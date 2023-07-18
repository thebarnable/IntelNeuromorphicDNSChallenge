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
from torch_mdct import MDCT, InverseMDCT


from lava.lib.dl import slayer
from audio_dataloader import DNSAudio
from snr import si_snr


def collate_fn(batch):
    noisy, clean, noise = [], [], []

    for sample in batch:
        noisy += [torch.FloatTensor(sample[0])]
        clean += [torch.FloatTensor(sample[1])]
        noise += [torch.FloatTensor(sample[2])]

    return torch.stack(noisy), torch.stack(clean), torch.stack(noise)


def stft_splitter(audio, window, n_fft=512):
    with torch.no_grad():
        audio_stft = torch.stft(audio,
                                n_fft=n_fft,
                                onesided=True,
                                return_complex=True, window=window)
        return audio_stft.abs(), audio_stft.angle()
    

def stft_mixer(stft_abs, stft_angle, window, n_fft=512):
    return torch.istft(torch.complex(stft_abs * torch.cos(stft_angle),
                                     stft_abs * torch.sin(stft_angle)),
                       n_fft=n_fft, onesided=True, window=window)
    

def calculate_filter_banks(n_fft, n_filter_banks, sample_rate, device):
    with torch.no_grad():
        f_l, f_u = 0, sample_rate/2
        def m_fn(f): return 1125*np.log(1+f/700)
        def m_inv_fn(m): return 700*(np.exp(m/1125) - 1)
        m_vals = [(m_fn(f_u) - m_fn(f_l))/(n_filter_banks+1) * i + m_fn(f_l)
                    for i in range(n_filter_banks+2)]
        f_vals = [np.floor((m_inv_fn(m) * (n_fft + 1))/sample_rate)
                    for m in m_vals]
        H_vals = np.zeros((n_filter_banks, n_fft//2+1), dtype = np.float32)

        map = {k: [] for k in range(n_fft//2+1)}
        for i in range(1, n_filter_banks+1):
            for k in range(n_fft//2+1):
                if f_vals[i-1] < k <= f_vals[i]:
                    H_vals[i-1, k] = (k-f_vals[i-1]) / \
                        (f_vals[i]-f_vals[i-1])
                    map[k].append(i-1)
                if f_vals[i] < k < f_vals[i+1]:
                    H_vals[i-1, k] = (f_vals[i+1]-k) / \
                        (f_vals[i+1]-f_vals[i])
                    map[k].append(i-1)

        return torch.from_numpy(H_vals).to(device), map


def reconstruct_wave_from_mfcc(n_fft, abs, phase, y_vals, y_vals_network, H_vals, map, inv_spec_transformation):
    x_vals = torch.clone(abs)

    for m in range(1, n_fft // 2):
        val = 0
        for i in range(len(map[m])):
            val += (y_vals_network[:, map[m][i], :] - y_vals[:, map[m][i], :] + H_vals[map[m][i], m] * abs[:, m, :]) / H_vals[map[m][i], m]
        
        x_vals[:, m, :] = val/len(map[m])
    x_vals[x_vals < 0] = 0


    return inv_spec_transformation(torch.complex(x_vals * torch.cos(phase),
                                x_vals * torch.sin(phase))), x_vals


def prob(p):
    rand = np.random.random()
    return rand < p

def vol_scaling(data_noisy, data_clean, sr):
    val = np.random.randint(-20, 0)
    transform = torchaudio.transforms.Vol(val, 'db')
    return transform(data_noisy), transform(data_clean)


def lowpass(data_noisy, data_clean, sr):
    cutoff = np.random.randint(3500, 4500)
    return torchaudio.functional.lowpass_biquad(data_noisy, sr, cutoff), torchaudio.functional.lowpass_biquad(data_clean, sr, cutoff)

def resample(data_noisy, data_clean, sr):
    new_freq = np.random.randint(int(0.75*sr/1000), int(1.25*sr/1000))
    if new_freq == sr//1000:
        return data_noisy, data_clean
    return torchaudio.functional.resample(data_noisy, sr//1000, new_freq), torchaudio.functional.resample(data_clean, sr//1000, new_freq)

def time_stretch(data_noisy, data_clean, sr):
    val = np.random.uniform(0.5,1.5)
    transform = torchaudio.transforms.TimeStretch()
    spec = torchaudio.transforms.Spectrogram(power = None)
    inv_spec = torchaudio.transforms.InverseSpectrogram()
    return inv_spec(transform(spec(data_noisy), val)), inv_spec(transform(spec(data_clean), val))

def clipping(data_noisy, data_clean, sr):
    val = np.random.uniform(0.8, 1) * torch.max(data_noisy)
    data_noisy[data_noisy > val] = val
    data_clean[data_clean > val] = val
    return data_noisy, data_clean


class Network(torch.nn.Module):
    def __init__(self, threshold=0.1, tau_grad=0.1, scale_grad=0.8, max_delay=64, out_delay=0, n_input = 257, n_hidden=512):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay

        sigma_params = {  # sigma-delta neuron parameters
            'threshold': threshold,   # delta unit threshold
            'tau_grad': tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad': scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad': False,  # trainable threshold
            'shared_param': True,   # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation': F.relu,  # activation function
        }

        self.input_quantizer = lambda x: slayer.utils.quantize(x, step=1 / 64)

        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Dense(
                sdnn_params, n_input, n_hidden, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Dense(
                sdnn_params, n_hidden, n_hidden, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Output(
                sdnn_params, n_hidden, n_input, weight_norm=False),
        ])

        self.blocks[0].pre_hook_fx = self.input_quantizer

        self.blocks[1].delay.max_delay = max_delay
        self.blocks[2].delay.max_delay = max_delay

    def forward(self, noisy):
        x = noisy - self.stft_mean

        for block in self.blocks:
            x = block(x)

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [
            b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def validate_gradients(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any()
                                       or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            self.zero_grad()

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


def nop_stats(dataloader, stats, sub_stats, print=True):
    t_st = datetime.now()
    for i, (noisy, clean, noise) in enumerate(dataloader):
        with torch.no_grad():
            noisy = noisy
            clean = clean

            score = si_snr(noisy, clean)
            sub_stats.correct_samples += torch.sum(score).item()
            sub_stats.num_samples += noisy.shape[0]

            processed = i * dataloader.batch_size
            total = len(dataloader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / dataloader.batch_size
            header_list = [f'Train: [{processed}/{total} '
                           f'({100.0 * processed / total:.0f}%)]']
            if print:
                stats.print(0, i, samples_sec, header=header_list)
                
                
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
    parser.add_argument("-n_mfcc", type=int, default=50, help="Number of MFCC components (only considered if transformation is mfcc or mse_loss_type is mfcc)")

    # ratio between input and hidden layer size
    parser.add_argument("-hidden_input_ratio", type=float, default=2.0, help="Ratio of hidden layer and input layer size (only considered if transformation=mfcc, for stft ratio is always 512/257)")

    # loss type
    parser.add_argument("-mse_loss_type", type=str, default = "stft", choices = ["stft", "mfcc"], help = "Which coefficients to use for MSE part of the loss function")

    parser.add_argument("-signal_quality_loss_type", type=str, default = "si-snr", choices = ["si-snr"], help = "Which metric to use to measure signal quality")

    # learn rate scheduler
    parser.add_argument("-lr_scheduler", type=str, default = None, choices = [None, "linear", "mult", "step"], help="Type of learn rate scheduler")
    
    # Data augmentation
    parser.add_argument("-data_aug_factor", type=float, default=0, help = "Intensity to apply data augmentation (between 0 and 1)")
    
    # Phase incorporation
    parser.add_argument("-phase_inc", type=float, default = 0, help = "Sampling ratio of phase information (0=phase is not used, 1=phase is fully incorporated)")



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

    if args.phase_inc > 0 and args.transformation != "stft":
        raise Exception("Phase incorporation only possible if transformation=stft!")
    

    print("Configuring network...")
    n_phase_features = int(args.phase_inc * (args.n_fft // 2 +1))
    if n_phase_features != 0:
        print("Using %d phase features" % n_phase_features)
        phase_feature_indices = torch.from_numpy(np.round(np.linspace(0, args.n_fft // 2, n_phase_features)).astype(int)).to(device)
    n_input = args.n_fft // 2 + 1 + n_phase_features if args.transformation == "stft" else 256 if args.transformation == "mdct" else args.n_mfcc
    n_hidden = args.n_fft + 2*n_phase_features if args.transformation == "stft" else 512 if args.transformation == "mdct" else int(args.n_mfcc * args.hidden_input_ratio)


    out_delay = args.out_delay
    if len(args.gpu) == 1:
        net = Network(args.threshold,
                      args.tau_grad,
                      args.scale_grad,
                      args.dmax,
                      args.out_delay,
                      n_input, n_hidden).to(device)
        module = net
    else:
        net = torch.nn.DataParallel(Network(args.threshold,
                                            args.tau_grad,
                                            args.scale_grad,
                                            args.dmax,
                                            args.out_delay,
                                            n_input, n_hidden).to(device),
                                    device_ids=args.gpu)
        module = net.module

    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-5)
    
    # Learn rate scheduler
    lr_scheduler = None
    if args.lr_scheduler == "linear":
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.03, verbose=True, total_iters=20)
    elif args.lr_scheduler == "mult":
        # only useful for lr=0.01: lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda ep : 1 if ep < 4 or ep > 10 else 0.75, verbose = True)
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lambda ep : 1 if ep < 13 or ep > 30 else 0.8, verbose = True)
    elif args.lr_scheduler == "step":    
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8 ,verbose=True)

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
        window_fn = lambda n : torch.hann_window(n, device=device)
    elif args.stft_window == "hamming":
        window_fn = lambda n : torch.hamming_window(n, device=device)
    elif args.stft_window == "bartlett":
        window_fn = lambda n : torch.bartlett_window(n, device=device)
    
    window = None
    if window_fn is not None:
        window = window_fn(args.n_fft)

    if args.transformation == "mfcc":
        if window_fn == None:
            window_fn_mfcc = lambda n : torch.hann_window(n, device=device)
        else:
            window_fn_mfcc = window_fn

        spec_transformation = torchaudio.transforms.Spectrogram(
            n_fft=args.n_fft, window_fn=window_fn_mfcc, power=None)
        inv_spec_transformation = torchaudio.transforms.InverseSpectrogram(
            n_fft=args.n_fft, window_fn=window_fn_mfcc)
        
        H_vals, map = calculate_filter_banks(args.n_fft, args.n_mfcc, 16000, device)
    
    
    if args.transformation == "mdct":
        if args.mse_loss_type == "stft":
            raise Exception("STFT as MSE loss type is not possible when transformation=mdct")
        
        mdct = MDCT(512).to(device)
        inv_mdct = InverseMDCT(512).to(device)
    
    if args.mse_loss_type == "mfcc":
        mse_loss_mfcc_transformation = torchaudio.transforms.MFCC(n_mfcc = args.n_mfcc, melkwargs = {"window_fn" : lambda n:torch.hann_window(n, device = device), 
                                                                                            "n_fft" : 512, "hop_length" : 128}).to(device)


    for epoch in range(args.epoch):
        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(train_loader):
            net.train()
            
            # Data augmentation
            if args.data_aug_factor > 0:
                #if prob(0.2 * args.data_aug_factor):
                #    noisy, clean = resample(noisy, clean, 16000)
                #    print("RESAMPLE")
                #if prob(0.1 * args.data_aug_factor):
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

            # USING STFT
            if args.transformation == "stft":
                noisy_abs, noisy_arg = stft_splitter(noisy, window,  args.n_fft)
                clean_abs, clean_arg = stft_splitter(clean, window, args.n_fft)
                
                if args.phase_inc > 0:
                    feat = torch.concat((noisy_abs, noisy_arg[:, phase_feature_indices, :]), dim=1)
                    result = net(feat)
                    denoised_abs = result[:, :-n_phase_features,:]
                    denoised_arg = torch.clone(noisy_arg)
                    denoised_arg[:,phase_feature_indices, :] = result[:, args.n_fft//2+1:, :]
                else:
                    denoised_abs = net(noisy_abs)
                    denoised_arg = noisy_arg


                denoised_arg = slayer.axon.delay(denoised_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

                
                idx_abs = torch.where(torch.isnan(denoised_abs))
                idx_arg = torch.where(torch.isnan(denoised_arg))

                clean_rec = stft_mixer(denoised_abs, denoised_arg, window, args.n_fft)
                
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

                clean_rec, denoised_abs = reconstruct_wave_from_mfcc(args.n_fft, noisy_abs, noisy_arg, filter_banked, torch.exp(denoised_mfcc), H_vals, map, inv_spec_transformation)
            
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
            
            loss = mse_loss + signal_loss
            
            if torch.isnan(loss):
                print("WARNING: NaN detected in loss! MSE loss: %.5f, Signal loss:%.5f, Total Loss: %.5f" % (mse_loss, signal_loss, loss))
            
            loss = torch.nan_to_num(loss, nan=0, posinf = 0, neginf = 0)

            assert torch.isnan(loss) == False

            optimizer.zero_grad()
            loss.backward()
            net.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            
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

            if i == 0:
                print("First iteration of training done without error...")

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
                    noisy_abs, noisy_arg = stft_splitter(noisy, window,  args.n_fft)
                    clean_abs, clean_arg = stft_splitter(clean, window, args.n_fft)
                    
                    if args.phase_inc > 0:
                        feat = torch.concat((noisy_abs, noisy_arg[:, phase_feature_indices, :]), dim=1)
                        result = net(feat)
                        denoised_abs = result[:, :-n_phase_features,:]
                        denoised_arg = torch.clone(noisy_arg)
                        denoised_arg[:,phase_feature_indices, :] = result[:, args.n_fft//2+1:, :]
                    else:
                        denoised_abs = net(noisy_abs)
                        denoised_arg = noisy_arg

                    denoised_arg = slayer.axon.delay(denoised_arg, out_delay)
                    clean_abs = slayer.axon.delay(clean_abs, out_delay)
                    clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

                    clean_rec = stft_mixer(denoised_abs, denoised_arg, window, args.n_fft)
                    
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

                    clean_rec, denoised_abs = reconstruct_wave_from_mfcc(args.n_fft, noisy_abs, noisy_arg, filter_banked, torch.exp(denoised_mfcc), H_vals, map, inv_spec_transformation)
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
                
                loss = mse_loss + signal_loss
                
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
