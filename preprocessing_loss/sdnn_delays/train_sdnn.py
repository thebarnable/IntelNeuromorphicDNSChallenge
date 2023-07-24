# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio.transforms
from time import time

from utility.torch_mdct import MDCT, InverseMDCT
from utility.quantization import *
from utility.helpers import *
from utility.audio_dataloader import DNSAudio
from utility.snr import si_snr
from utility.networks import Network, ConvNetwork

from lava.lib.dl import slayer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu',
                        type=int,
                        default=[0],
                        help='which gpu(s) to use', nargs='+')
    parser.add_argument('-b',
                        type=int,
                        default=32,
                        help='batch size for dataloader')
    parser.add_argument('-lr',
                        type=float,
                        default=0.01,
                        help='initial learning rate')
    parser.add_argument('-lam',
                        type=float,
                        default=0.001,
                        help='lagrangian factor')
    parser.add_argument('-threshold',
                        type=float,
                        default=0.1,
                        help='neuron threshold')
    parser.add_argument('-tau_grad',
                        type=float,
                        default=0.1,
                        help='surrogate gradient time constant')
    parser.add_argument('-scale_grad',
                        type=float,
                        default=0.8,
                        help='surrogate gradient scale')
    parser.add_argument('-n_fft',
                        type=int,
                        default=512,
                        help='number of FFT specturm, hop is n_fft // 4')
    parser.add_argument('-dmax',
                        type=int,
                        default=64,
                        help='maximum axonal delay')
    parser.add_argument('-out_delay',
                        type=int,
                        default=0,
                        help='prediction output delay (multiple of 128)')
    parser.add_argument('-clip',
                        type=float,
                        default=10,
                        help='gradient clipping limit')
    parser.add_argument('-exp',
                        type=str,
                        default='',
                        help='experiment differentiater string')
    parser.add_argument('-seed',
                        type=int,
                        default=None,
                        help='random seed of the experiment')
    parser.add_argument('-epoch',
                        type=int,
                        default=50,
                        help='number of epochs to run')
    parser.add_argument('-path',
                        type=str,
                        default='../../data/MicrosoftDNS_4_ICASSP/',
                        help='dataset path')
    parser.add_argument('-out',
                        type=str,
                        default='runs/',
                        help='results path')

    ##########################################
    # Additional ENCODING/DECODING arguments #
    ##########################################

    # STFT arguments
    parser.add_argument("-stft_window", type=str, choices=["hann", "hamming", "gaussian", "bartlett", "rectangle"],
                        default="rectangle", help="window function for STFT (considered if transformation=stft or transformation=mfcc)")

    # transformation type
    parser.add_argument("-transformation", type=str, choices=["stft", "mfcc", "mdct"],
                        default="stft", help="transformation that is applied as encoding/decoding")

    # number of mfcc components
    parser.add_argument("-n_mfcc", type=int, default=50,
                        help="Number of MFCC components (only considered if transformation is mfcc or mse_loss_type is mfcc)")

    # ratio between input and hidden layer size
    #parser.add_argument("-hidden_input_ratio", type=float, default=2.0,
    #                    help="Ratio of hidden layer and input layer size (only considered if transformation=mfcc, for stft ratio is always 512/257)")
    
    # number of hidden neurons
    parser.add_argument("-n_hidden", type=int, default=512, help="Number of hidden neurons.")

    # loss type
    parser.add_argument("-mse_loss_type", type=str, default="stft", choices=[
                        "stft", "mfcc"], help="Which coefficients to use for MSE part of the loss function")

    parser.add_argument("-signal_quality_loss_type", type=str, default="si-snr",
                        choices=["si-snr"], help="Which metric to use to measure signal quality")

    # learn rate scheduler
    parser.add_argument("-lr_scheduler", type=str, default=None, choices=[
                        None, "linear", "mult", "step", "multistep", "mult2"], help="Type of learn rate scheduler")

    # Data augmentation
    parser.add_argument("-data_aug_factor", type=float, default=0,
                        help="Intensity to apply data augmentation (between 0 and 1)")

    # Phase incorporation
    parser.add_argument("-phase_inc", type=float, default=0,
                        help="Sampling ratio of phase information (0=phase is not used, 1=phase is fully incorporated)")

    # Binarization
    parser.add_argument('-binarization',
                        type=bool,
                        default=False,
                        help='Apply binarization')
    
    # Ternarization
    parser.add_argument('-ternarization',
                        type=bool,
                        default=False,
                        help='apply ternarization')
    
    # Quantization
    parser.add_argument('-quantization',
                        type=bool,
                        default=False,
                        help='apply quantization')
    parser.add_argument('-bits',
                        type=int,
                        default=8,
                        help='quantization bits')

    # Network Architecture
    parser.add_argument("-architecture", type=str, default="baseline", choices=["baseline", "conv"], help="Network architecture to use.")
    
    # Number of hidden layers
    parser.add_argument("-n_layers", type=int, default=2, help="Number of hidden layers.")
    
    # Kernel size for the conv network
    parser.add_argument("-kernel_size", type=int, default=5, help="Kernel size. Only considered if architecture=conv.")
    
    # Batch Normalization
    parser.add_argument("-batch_norm", type=bool, default = False, help="Enable neuron batch normalization.")
    
    # Batch Normalization
    parser.add_argument("-dropout", type=bool, default = False, help="Enable neuron dropout.")

    # Debug Weights
    parser.add_argument("-debug_weights", type=bool, default = False, help="Enable debugging of weights")
    
    # Additional loss
    parser.add_argument("-loss_non_original", type=bool, default=False, help="Add an addition loss to prevent network to learn output=input")
    

    args = parser.parse_args()

    identifier = args.exp
    if args.seed is not None:
        torch.manual_seed(args.seed)
        identifier += '_{}{}'.format(args.optim, args.seed)

    trained_folder = 'Trained' + identifier
    logs_folder = 'Logs' + identifier
    print(trained_folder)
    writer = SummaryWriter(args.out + identifier)

    os.makedirs(trained_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    lam = args.lam

    print('Using GPUs {}'.format(args.gpu))
    device = torch.device('cuda:{}'.format(args.gpu[0]))

    # quantization observer to determine parameters
    obs = torch.ao.quantization.observer.MinMaxObserver(
        quant_min=0, quant_max=2**args.bits - 1)
    obs.to(device=device)

    if args.phase_inc > 0 and args.transformation != "stft":
        raise Exception(
            "Phase incorporation only possible if transformation=stft!")

    print("Configuring network...")
    n_phase_features = int(args.phase_inc * (args.n_fft // 2 + 1))
    if n_phase_features != 0:
        print("Using %d phase features" % n_phase_features)
        phase_feature_indices = torch.from_numpy(np.round(np.linspace(
            0, args.n_fft // 2, n_phase_features)).astype(int)).to(device)
    n_input = args.n_fft // 2 + 1 + \
        n_phase_features if args.transformation == "stft" else 256 if args.transformation == "mdct" else args.n_mfcc
    #n_hidden = args.n_fft + 2 * \
    #    n_phase_features if args.transformation == "stft" else 512 if args.transformation == "mdct" else int(
    #        args.n_mfcc * args.hidden_input_ratio)
    n_hidden = args.n_hidden

    out_delay = args.out_delay
    
    if args.architecture == "baseline":
        net = Network(args.threshold, args.tau_grad, args.scale_grad, args.dmax,
                      args.out_delay, n_input, n_hidden, args.n_layers, args.batch_norm, args.dropout, args.binarization, args.ternarization,
                      args.quantization, obs).to(device)
    elif args.architecture == "conv":
        if args.quantization or args.binarization or args.ternarization:
            raise Exception("Quantization, Binarization and Ternarization is not implemented for the Conv Network") 
        
        net = ConvNetwork(args.threshold, args.tau_grad, args.scale_grad, args.dmax,
                      args.out_delay, n_input, n_hidden, args.n_layers, args.batch_norm, args.dropout, args.kernel_size).to(device)
    
    
    if len(args.gpu) == 1:
        module = net
    else:
        net = torch.nn.DataParallel(net, device_ids=args.gpu)
        module = net.module

    # Define optimizer module.
    if args.architecture == "baseline":
        optimizer = torch.optim.RAdam(net.parameters(),
                                    lr=args.lr,
                                    weight_decay=1e-5 if not args.binarization else 0)
    elif args.architecture == "conv":
        optimizer = torch.optim.RAdam(net.parameters(),
                                    lr=args.lr,
                                    weight_decay=1e-3)

    # Learn rate scheduler
    lr_scheduler = None
    if args.lr_scheduler == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1, end_factor=0.03, verbose=True, total_iters=20)
    elif args.lr_scheduler == "mult":
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lambda ep: 1 if ep < 13 or ep > 30 else 0.8, verbose=True)
    elif args.lr_scheduler == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=2, gamma=0.8, verbose=True)
    elif args.lr_scheduler == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20,35,50], gamma=0.1, verbose=True)
    elif args.lr_scheduler == "mult2":
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer, lambda ep: 1 if ep < 5 or ep >= 60 else 0.85, verbose=True)

    train_set = DNSAudio(root=args.path + 'training_set/')
    validation_set = DNSAudio(root=args.path + 'validation_set/')

    print("Configuring network finished")
    print("Loading data sets")

    train_loader = DataLoader(train_set,
                              batch_size=args.b,
                              shuffle=True,
                              collate_fn=collate_fn,
                              num_workers=4,
                              pin_memory=True)
    validation_loader = DataLoader(validation_set,
                                   batch_size=args.b,
                                   shuffle=True,
                                   collate_fn=collate_fn,
                                   num_workers=4,
                                   pin_memory=True)

    print("Loading data sets finished")

    base_stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                            accuracy_unit='dB')

    # print()
    # print('Base Statistics')
    # nop_stats(train_loader, base_stats, base_stats.training)
    # nop_stats(validation_loader, base_stats, base_stats.validation)
    # print()

    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                       accuracy_unit='dB')

    window_fn = None  # treated as rectangular window function
    if args.stft_window == "hann":
        def window_fn(n): return torch.hann_window(n, device=device)
    elif args.stft_window == "hamming":
        def window_fn(n): return torch.hamming_window(n, device=device)
    elif args.stft_window == "bartlett":
        def window_fn(n): return torch.bartlett_window(n, device=device)

    window = None
    if window_fn is not None:
        window = window_fn(args.n_fft)

    if args.transformation == "mfcc":
        if window_fn == None:
            def window_fn_mfcc(n): return torch.hann_window(n, device=device)
        else:
            window_fn_mfcc = window_fn

        spec_transformation = torchaudio.transforms.Spectrogram(
            n_fft=args.n_fft, window_fn=window_fn_mfcc, power=None)
        inv_spec_transformation = torchaudio.transforms.InverseSpectrogram(
            n_fft=args.n_fft, window_fn=window_fn_mfcc)

        H_vals, map = calculate_filter_banks(
            args.n_fft, args.n_mfcc, 16000, device)

    if args.transformation == "mdct":
        if args.mse_loss_type == "stft":
            raise Exception(
                "STFT as MSE loss type is not possible when transformation=mdct")

        mdct = MDCT(512).to(device)
        inv_mdct = InverseMDCT(512).to(device)

    if args.mse_loss_type == "mfcc":
        mse_loss_mfcc_transformation = torchaudio.transforms.MFCC(n_mfcc=args.n_mfcc, melkwargs={"window_fn": lambda n: torch.hann_window(n, device=device),
                                                                                                 "n_fft": 512, "hop_length": 128}).to(device)

    for epoch in range(args.epoch):
        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(train_loader):
            net.train()

            # Data augmentation
            if args.data_aug_factor > 0:
                # if prob(0.2 * args.data_aug_factor):
                #    noisy, clean = resample(noisy, clean, 16000)
                #    print("RESAMPLE")
                # if prob(0.1 * args.data_aug_factor):
                #    noisy, clean = time_stretch(noisy, clean, 16000)
                #    print("TIME STR")
                if prob(0.1 * args.data_aug_factor):
                    noisy, clean = clipping(noisy, clean, 16000)
                if prob(0.1 * args.data_aug_factor):
                    noisy, clean = lowpass(noisy, clean, 16000)
                if prob(args.data_aug_factor):
                    noisy, clean = vol_scaling(noisy, clean, 16000)

            noisy = noisy.to(device)
            clean = clean.to(device)

            if args.debug_weights and i % 400 == 0:
                torch.set_printoptions(profile="full", linewidth=1000)
                with open(trained_folder + '/weights_before_%d_%d.txt' % (epoch, i), 'w+') as f:
                    for name, val in net.named_parameters():
                        if "weight" in name:
                            f.write(name+" : "+str(val.data.squeeze())+"\n")
                torch.set_printoptions(profile="default")

            # USING STFT
            if args.transformation == "stft":
                noisy_abs, noisy_arg = stft_splitter(
                    noisy, window,  args.n_fft)
                clean_abs, clean_arg = stft_splitter(clean, window, args.n_fft)

                if args.phase_inc > 0:
                    feat = torch.concat(
                        (noisy_abs, noisy_arg[:, phase_feature_indices, :]), dim=1)
                    result = net(feat)
                    denoised_abs = result[:, :-n_phase_features, :]
                    denoised_arg = torch.clone(noisy_arg)
                    denoised_arg[:, phase_feature_indices,
                                 :] = result[:, args.n_fft//2+1:, :]
                else:
                    denoised_abs = net(noisy_abs)
                    denoised_arg = noisy_arg

                denoised_arg = slayer.axon.delay(denoised_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

                clean_rec = stft_mixer(
                    denoised_abs, denoised_arg, window, args.n_fft)

            # USING MFCC with approximated inversion
            elif args.transformation == "mfcc":
                spectogram_noisy = spec_transformation(noisy)
                spectogram_clean = spec_transformation(clean)

                noisy_abs, noisy_arg = spectogram_noisy.abs(), spectogram_noisy.angle()
                clean_abs, clean_arg = spectogram_clean.abs(), spectogram_clean.angle()

                filter_banked = torch.matmul(H_vals, noisy_abs) + 1e-10
                mfcc = torch.log(filter_banked)

                denoised_mfcc = net(mfcc)

                noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

                clean_rec, denoised_abs = reconstruct_wave_from_mfcc(
                    args.n_fft, noisy_abs, noisy_arg, filter_banked, torch.exp(denoised_mfcc), H_vals, map, inv_spec_transformation)

            elif args.transformation == "mdct":
                mdct_features = mdct(noisy)
                denoised_mdct = net(mdct_features)
                clean_rec = inv_mdct(denoised_mdct)[:, :480000]

            if args.mse_loss_type == "stft":
                mse_loss = lam * F.mse_loss(denoised_abs, clean_abs)
            elif args.mse_loss_type == "mfcc":
                mfcc_clean = mse_loss_mfcc_transformation(clean)
                mfcc_denoised = mse_loss_mfcc_transformation(clean_rec)
                mse_loss = lam * F.mse_loss(mfcc_clean, mfcc_denoised)

            if args.signal_quality_loss_type == "si-snr":
                score = si_snr(clean_rec, clean[:, :clean_rec.size(dim=1)])
                signal_loss = (100 - torch.mean(score))
                
            loss_add = 0
            if args.loss_non_original:
                loss_add = 1e-2 * torch.abs((0.5 - F.mse_loss(denoised_abs, noisy_abs)))

            loss = mse_loss + signal_loss + loss_add
            print(mse_loss.item(), signal_loss.item())

            if torch.isnan(loss):
                print("WARNING: NaN detected in loss! MSE loss: %.5f, Signal loss:%.5f, Total Loss: %.5f" % (
                    mse_loss, signal_loss, loss))

            loss = torch.nan_to_num(loss, nan=0, posinf=0, neginf=0)
            

            assert torch.isnan(loss) == False

            optimizer.zero_grad()
            loss.backward()
            net.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            
            if args.binarization:
                for m in net.modules():
                    if isinstance(m, DenseBinary):
                        m.synapse.load_non_binary_weights()
            elif args.ternarization:
                for m in net.modules():
                    if isinstance(m, DenseTernary):
                        m.synapse.load_non_ternary_weights()
                
            
            optimizer.step()
            
            if args.binarization:
                for m in net.modules():
                    if isinstance(m, DenseBinary):
                        m.synapse.clamp()
                        m.synapse.save_non_binary_weights()
            elif args.ternarization:
                for m in net.modules():
                    if isinstance(m, DenseTernary):
                        m.synapse.clamp()
                        m.synapse.save_non_ternary_weights()

            if i < 10:
                net.grad_flow(path=trained_folder + '/')

            if args.signal_quality_loss_type == "si-snr":
                if torch.isnan(score).any():
                    score[torch.isnan(score)] = 0

                stats.training.correct_samples += torch.sum(score).item()
            stats.training.loss_sum += loss.item()
            stats.training.num_samples += noisy.shape[0]

            processed = i * train_loader.batch_size
            total = len(train_loader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
            header_list = [f'Train: [{processed}/{total} '
                           f'({100.0 * processed / total:.0f}%)]']
            stats.print(epoch, i, samples_sec, header=header_list)

            if args.debug_weights and i % 400 == 0:
                torch.set_printoptions(profile="full", linewidth=1000)
                with open(trained_folder + '/weights_after_%d_%d.txt' % (epoch, i), 'w+') as f:
                    for name, val in net.named_parameters():
                        if "weight" in name:
                            f.write(name+" : "+str(val.data.squeeze())+"\n")
                torch.set_printoptions(profile="default")
                
        if lr_scheduler is not None:
            lr_scheduler.step()

        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(validation_loader):
            net.eval()

            with torch.no_grad():
                noisy = noisy.to(device)
                clean = clean.to(device)

                # USING STFT
                if args.transformation == "stft":
                    noisy_abs, noisy_arg = stft_splitter(
                        noisy, window,  args.n_fft)
                    clean_abs, clean_arg = stft_splitter(
                        clean, window, args.n_fft)

                    if args.phase_inc > 0:
                        feat = torch.concat(
                            (noisy_abs, noisy_arg[:, phase_feature_indices, :]), dim=1)
                        result = net(feat)
                        denoised_abs = result[:, :-n_phase_features, :]
                        denoised_arg = torch.clone(noisy_arg)
                        denoised_arg[:, phase_feature_indices,
                                     :] = result[:, args.n_fft//2+1:, :]
                    else:
                        denoised_abs = net(noisy_abs)
                        denoised_arg = noisy_arg

                    denoised_arg = slayer.axon.delay(denoised_arg, out_delay)
                    clean_abs = slayer.axon.delay(clean_abs, out_delay)
                    clean = slayer.axon.delay(
                        clean, args.n_fft // 4 * out_delay)

                    clean_rec = stft_mixer(
                        denoised_abs, denoised_arg, window, args.n_fft)

                # USING MFCC with approximated inversion
                elif args.transformation == "mfcc":
                    spectogram_noisy = spec_transformation(noisy)
                    spectogram_clean = spec_transformation(clean)

                    noisy_abs, noisy_arg = spectogram_noisy.abs(), spectogram_noisy.angle()
                    clean_abs, clean_arg = spectogram_clean.abs(), spectogram_clean.angle()

                    filter_banked = torch.matmul(H_vals, noisy_abs) + 1e-10
                    mfcc = torch.log(filter_banked)

                    denoised_mfcc = net(mfcc)

                    noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                    clean_abs = slayer.axon.delay(clean_abs, out_delay)
                    clean = slayer.axon.delay(
                        clean, args.n_fft // 4 * out_delay)

                    clean_rec, denoised_abs = reconstruct_wave_from_mfcc(
                        args.n_fft, noisy_abs, noisy_arg, filter_banked, torch.exp(denoised_mfcc), H_vals, map, inv_spec_transformation)
                # USING MDCT
                elif args.transformation == "mdct":
                    mdct_features = mdct(noisy)
                    denoised_mdct = net(mdct_features)
                    clean_rec = inv_mdct(denoised_mdct)[:, :480000]

                if args.mse_loss_type == "stft":
                    mse_loss = lam * F.mse_loss(denoised_abs, clean_abs)
                elif args.mse_loss_type == "mfcc":
                    mfcc_clean = mse_loss_mfcc_transformation(clean)
                    mfcc_denoised = mse_loss_mfcc_transformation(clean_rec)
                    mse_loss = lam * F.mse_loss(mfcc_clean, mfcc_denoised)

                if args.signal_quality_loss_type == "si-snr":
                    score = si_snr(clean_rec, clean)
                    signal_loss = (100 - torch.mean(score))
                    
                loss_add = 0
                if args.loss_non_original:
                    loss_add = 1e-2 * torch.abs((0.5 - F.mse_loss(denoised_abs, noisy_abs)))

                loss = mse_loss + signal_loss + loss_add

                if args.signal_quality_loss_type == "si-snr":
                    stats.validation.correct_samples += torch.sum(score).item()
                stats.validation.loss_sum += loss.item()
                stats.validation.num_samples += noisy.shape[0]

                processed = i * validation_loader.batch_size
                total = len(validation_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / \
                    (i + 1) / validation_loader.batch_size
                header_list = [f'Valid: [{processed}/{total} '
                               f'({100.0 * processed / total:.0f}%)]']
                stats.print(epoch, i, samples_sec, header=header_list)

        writer.add_scalar('Loss/train', stats.training.loss, epoch)
        writer.add_scalar('Loss/valid', stats.validation.loss, epoch)
        writer.add_scalar('SI-SNR/train', stats.training.accuracy, epoch)
        writer.add_scalar('SI-SNR/valid', stats.validation.accuracy, epoch)

        stats.update()
        stats.plot(path=trained_folder + '/')
        if stats.validation.best_accuracy is True:
            torch.save(module.state_dict(), trained_folder + '/network.pt')
        stats.save(trained_folder + '/')

    net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    net.export_hdf5(trained_folder + '/network.net')

    params_dict = {}
    for key, val in args._get_kwargs():
        params_dict[key] = str(val)
    writer.add_hparams(params_dict, {'SI-SNR': stats.validation.max_accuracy})
    writer.flush()
    writer.close()
