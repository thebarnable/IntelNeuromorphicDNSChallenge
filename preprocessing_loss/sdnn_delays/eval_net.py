import numpy as np
from datetime import datetime
import yaml
import librosa
import torchaudio
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
import argparse
from lava.lib.dl import slayer

from utility.audio_dataloader import DNSAudio
from utility.snr import si_snr
from utility.dnsmos import DNSMOS
from utility.torch_mdct import MDCT, InverseMDCT
from utility.networks import Network, ConvNetwork
from utility.binarization import *

import os
from time import time

from utility.helpers import *


import onnxruntime as ort

print(ort.get_available_providers())


class InferenceNet(Network):
    def forward(self, noisy):
        x = noisy - self.stft_mean

        counts = []
        for block in self.blocks:
            x = block(x)
            count = torch.mean((torch.abs(x) > 0).to(x.dtype))
            counts.append(count.item())

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask, torch.tensor(counts)
    
class InferenceConvNet(ConvNetwork):
    def forward(self, noisy):
        x = noisy - self.stft_mean
        x = torch.unsqueeze(x, dim=2)
        x = torch.unsqueeze(x, dim=4)

        counts = []
        for block in self.blocks:
            x = block(x)
            count = torch.mean((torch.abs(x) > 0).to(x.dtype))
            counts.append(count.item())
            
        x = torch.squeeze(x)

        mask = torch.relu(x + 1)
        return slayer.axon.delay(noisy, self.out_delay) * mask, torch.tensor(counts)

def print_stats(file, str):
    with open(file, 'a') as f:
        f.write(str + os.linesep)
    print(str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-exp", type=str, default="baseline", help="Experiment identifier")
    parser.add_argument("-gpu", type=int, default=0, help="GPU to use")
    parser.add_argument("-no_train", action="store_false", dest="train", help="Skip calculation of in-sample statistics")
    parser.add_argument("-train_size", type=int, default=60000, help="Number of train samples to consider")
    parser.add_argument("-valid_size", type=int, default=60000, help="Number of validation samples to consider")
    parser.add_argument("-eval_out", type=str, default=None, help="Folder to save output files (example audio files, statistic files) into; default=eval_results/<exp>/")
    
    cur_args = parser.parse_args()
    
    if cur_args.eval_out is None:
        eval_out = "eval_results/"+cur_args.exp+"/"
    else:
        eval_out = cur_args.eval_out
    
    os.makedirs(eval_out, exist_ok=True)
    stats_file = eval_out+'stats_%s.txt' % cur_args.exp
    
    trained_folder = 'Trained' + cur_args.exp
    args = yaml.safe_load(open(trained_folder + '/args.txt', 'rt'))
    if 'out_delay' not in args.keys():
        args['out_delay'] = 0
    if 'n_fft' not in args.keys():
        args['n_fft'] = 512
        
    device = torch.device('cuda:'+str(cur_args.gpu))
    
    root = args['path']
    out_delay = args['out_delay']
    n_fft = args['n_fft']
    win_length = n_fft
    hop_length = n_fft // 4
    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR', accuracy_unit='dB')

    window_fn = None
    if "stft_window" in args.keys():
        if args["stft_window"] == "hann":
            window_fn = lambda n : torch.hann_window(n, device=device)
        elif args["stft_window"] == "hamming":
            window_fn = lambda n : torch.hamming_window(n, device=device)
        elif args["stft_window"] == "bartlett":
            window_fn = lambda n : torch.bartlett_window(n, device=device)

    window = None
    if window_fn is not None:
        window = window_fn(n_fft)
        
    transformation = "stft"
    if "transformation" in args.keys():
        transformation = args["transformation"]
    
    n_mfcc = 50
    if "n_mfcc" in args.keys():
        n_mfcc = args["n_mfcc"]
    
    hidden_input_ratio = 2.0
    if "hidden_input_ratio" in args.keys():
        hidden_input_ratio = args["hidden_input_ratio"]
    
    mse_loss_type = "stft"
    if "mse_loss_type" in args.keys():
        mse_loss_type = args["mse_loss_type"]
    
    phase_inc = 0
    if "phase_inc" in args.keys():
        phase_inc = args["phase_inc"]
        
    binarization = False
    if "binarization" in args.keys():
        binarization = args["binarization"]
        
    ternarization = False
    if "ternarization" in args.keys():
        ternarization = args["ternarization"] 
        
    quantization = False
    if "quantization" in args.keys():
        quantization = args["quantization"] 
    
    q_bits = 8
    if "bits" in args.keys():
        q_bits = args["bits"]
        
    arch = "baseline"
    if "architecture" in args.keys():
        arch = args["architecture"]
        
    n_layers = 2
    if "n_layers" in args.keys():
        n_layers = args["n_layers"]
    
    kernel_size = 5
    if "kernel_size" in args.keys():
        kernel_size = args[kernel_size]
        
    
    train_set = DNSAudio(root=root + 'training_set/')
    validation_set = DNSAudio(root=root + 'validation_set/')
    
    train_sampler = RandomSampler(train_set, replacement=False, num_samples=cur_args.train_size)
    valid_sampler = RandomSampler(train_set, replacement=False, num_samples=cur_args.valid_size)

    train_loader = DataLoader(train_set,
                            batch_size=32,
                            sampler=train_sampler,
                            collate_fn=collate_fn,
                            num_workers=4,
                            pin_memory=True)

    validation_loader = DataLoader(validation_set,
                                batch_size=32,
                                sampler=valid_sampler,
                                collate_fn=collate_fn,
                                num_workers=4,
                                pin_memory=True)
    
    
    if transformation == "mfcc":
        if window_fn == None:
            window_fn_mfcc = lambda n : torch.hann_window(n, device=device)
        else:
            window_fn_mfcc = window_fn

        spec_transformation = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, window_fn=window_fn_mfcc, power=None)
        inv_spec_transformation = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft, window_fn=window_fn_mfcc)
        
        H_vals, map = calculate_filter_banks(n_fft, n_mfcc, 16000, device)

    if transformation == "mdct":
        if mse_loss_type == "stft":
            raise Exception("STFT as MSE loss type is not possible when transformation=mdct")
        
        mdct = MDCT(512).to(device)
        inv_mdct = InverseMDCT(512).to(device)
    
    if mse_loss_type == "mfcc":
        mse_loss_mfcc_transformation = torchaudio.transforms.MFCC(n_mfcc = n_mfcc, melkwargs = {"window_fn" : lambda n:torch.hann_window(n, device = device), 
                                                                                            "n_fft" : 512, "hop_length" : 128}).to(device)

    obs = torch.ao.quantization.observer.MinMaxObserver(
        quant_min=0, quant_max=2**q_bits - 1)
    obs.to(device=device)

    n_phase_features = int(phase_inc * (n_fft // 2 + 1))
    if n_phase_features != 0:
        print("Using %d phase features" % n_phase_features)
        phase_feature_indices = torch.from_numpy(np.round(np.linspace(0, n_fft // 2, n_phase_features)).astype(int)).to(device)
    n_input = n_fft // 2 + 1 + n_phase_features if transformation == "stft" else 256 if transformation == "mdct" else n_mfcc
    n_hidden = n_fft + 2*n_phase_features if transformation == "stft" else 512 if transformation == "mdct" else int(n_mfcc * hidden_input_ratio)

    if arch == "baseline":
        net = InferenceNet(args['threshold'],
                    args['tau_grad'],
                    args['scale_grad'],
                    args['dmax'],
                    args['out_delay'],
                    n_input, n_hidden, n_layers,
                    binarization, ternarization, quantization, obs).to(device)
    elif arch == "conv":
        net = InferenceConvNet(args['threshold'],
                    args['tau_grad'],
                    args['scale_grad'],
                    args['dmax'],
                    args['out_delay'],
                    n_input, n_hidden, n_layers, kernel_size)
    
    noisy, clean, noise, metadata = train_set[0]
    noisy = torch.unsqueeze(torch.FloatTensor(noisy), dim=0).to(device)
    noisy_abs, noisy_arg = stft_splitter(noisy, window, n_fft)
    net(noisy_abs)
    net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    
    print("NETWORK:")
    print(net)
    
    if binarization:
        for m in net.modules():
            if isinstance(m, DenseBinary):
                m.synapse.binarize()
        
    
    ############################
    ### IN-SAMPLE STATISTICS ###
    ############################
    
    
    dnsmos = DNSMOS(providers=["CUDAExecutionProvider"])
    
    if cur_args.train:
        print_stats(stats_file, "IN-SAMPLE STATISTICS (n=%d)\n" % cur_args.train_size)
        dnsmos_noisy = np.zeros(3)
        dnsmos_clean = np.zeros(3)
        dnsmos_noise = np.zeros(3)
        dnsmos_cleaned  = np.zeros(3)
        train_event_counts = []

        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(validation_loader):
            net.eval()
            with torch.no_grad():
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                # USING STFT
                if transformation == "stft":
                    noisy_abs, noisy_arg = stft_splitter(noisy, window,  n_fft)
                    clean_abs, clean_arg = stft_splitter(clean, window, n_fft)
                    
                    if phase_inc > 0:
                        feat = torch.concat((noisy_abs, noisy_arg[:, phase_feature_indices, :]), dim=1)
                        result, count = net(feat)
                        denoised_abs = result[:, :-n_phase_features,:]
                        denoised_arg = torch.clone(noisy_arg)
                        denoised_arg[:,phase_feature_indices, :] = result[:, n_fft//2+1:, :]
                    else:
                        denoised_abs, count = net(noisy_abs)
                        denoised_arg = noisy_arg


                    denoised_arg = slayer.axon.delay(denoised_arg, out_delay)
                    clean_abs = slayer.axon.delay(clean_abs, out_delay)
                    clean = slayer.axon.delay(clean, n_fft // 4 * out_delay)

                    
                    idx_abs = torch.where(torch.isnan(denoised_abs))
                    idx_arg = torch.where(torch.isnan(denoised_arg))

                    clean_rec = stft_mixer(denoised_abs, denoised_arg, window, n_fft)
                    
                # USING MFCC with approximated inversion
                elif transformation == "mfcc":
                    spectogram_noisy = spec_transformation(noisy)
                    spectogram_clean = spec_transformation(clean)

                    noisy_abs, noisy_arg = spectogram_noisy.abs(), spectogram_noisy.angle()
                    clean_abs, clean_arg = spectogram_clean.abs(), spectogram_clean.angle()

                    filter_banked = torch.matmul(H_vals, noisy_abs) + 1e-10
                    mfcc = torch.log(filter_banked)
                    
                    denoised_mfcc, count = net(mfcc)

                    noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                    clean_abs = slayer.axon.delay(clean_abs, out_delay)
                    clean = slayer.axon.delay(clean, n_fft // 4 * out_delay)

                    clean_rec, denoised_abs = reconstruct_wave_from_mfcc(n_fft, noisy_abs, noisy_arg, filter_banked, torch.exp(denoised_mfcc), H_vals, map, inv_spec_transformation)
                
                elif transformation == "mdct":
                    mdct_features = mdct(noisy)
                    denoised_mdct, count = net(mdct_features)
                    clean_rec = inv_mdct(denoised_mdct)[:, :480000]
                         
                train_event_counts.append(count.cpu().data.numpy())

                if mse_loss_type == "stft":
                    loss = F.mse_loss(denoised_abs, clean_abs)
                elif mse_loss_type == "mfcc":
                    mfcc_clean = mse_loss_mfcc_transformation(clean)
                    mfcc_denoised = mse_loss_mfcc_transformation(clean_rec)
                    loss = F.mse_loss(mfcc_clean, mfcc_denoised)
            
            
                score = si_snr(clean_rec, clean)

                dnsmos_noisy += np.sum(dnsmos(noisy.cpu().data.numpy()), axis=0)
                dnsmos_clean += np.sum(dnsmos(clean.cpu().data.numpy()), axis=0)
                dnsmos_noise += np.sum(dnsmos(noise.cpu().data.numpy()), axis=0)
                dnsmos_cleaned += np.sum(dnsmos(clean_rec.cpu().data.numpy()), axis=0)
                

                stats.training.correct_samples += torch.sum(score).item()
                stats.training.loss_sum += loss.item()
                stats.training.num_samples += noisy.shape[0]

                processed = i * train_loader.batch_size
                total = len(train_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
                header_list = [f'Train: [{processed}/{total} '
                                f'({100.0 * processed / total:.0f}%)]']
                header_list.append(f'Event rate: {[c.item() for c in count]}')
                print(f'\r{header_list[0]}', end='')

        dnsmos_clean /= cur_args.train_size
        dnsmos_noisy /= cur_args.train_size
        dnsmos_noise /= cur_args.train_size
        dnsmos_cleaned /= cur_args.train_size

        print()
        stats.print(0, i, samples_sec, header=header_list)
        print_stats(stats_file, 'Avg DNSMOS clean   [ovrl, sig, bak]: '+str(dnsmos_clean))
        print_stats(stats_file,'Avg DNSMOS noisy   [ovrl, sig, bak]: ' +str(dnsmos_noisy))
        print_stats(stats_file,'Avg DNSMOS noise   [ovrl, sig, bak]: '+str(dnsmos_noise))
        print_stats(stats_file,'Avg DNSMOS cleaned [ovrl, sig, bak]: '+str(dnsmos_cleaned))

        mean_events = np.mean(train_event_counts, axis=0)

        neuronops = []
        for block in net.blocks[:-1]:
            neuronops.append(np.prod(block.neuron.shape))

        synops = []
        for events, block in zip(mean_events, net.blocks[1:]):
            synops.append(events * np.prod(block.synapse.shape))
        print_stats(stats_file, f'SynOPS: {synops}')
        print_stats(stats_file, f'Total SynOPS: {sum(synops)}')
        print_stats(stats_file, f'Total NeuronOPS: {sum(neuronops)}')
        print_stats(stats_file, f'Time-step per sample: {noisy_abs.shape[-1]}')
        
        
        # Save in-sample example audio
        torchaudio.save(eval_out+'train_example_noisy.wav', noisy[0:1, :].cpu(), 16000)
        torchaudio.save(eval_out+'train_example_clean.wav', clean[0:1, :].cpu(), 16000)
        torchaudio.save(eval_out+'train_example_clean_rec.wav', clean_rec[0:1, :].cpu(), 16000)
        
    #############################
    ### VALIDATION STATISTICS ###
    #############################
    
    print_stats(stats_file, "\nVALIDATION STATISTICS (n=%d)\n" % cur_args.valid_size)
        
    dnsmos_noisy = np.zeros(3)
    dnsmos_clean = np.zeros(3)
    dnsmos_noise = np.zeros(3)
    dnsmos_cleaned  = np.zeros(3)
    valid_event_counts = []

    t_st = datetime.now()
    for i, (noisy, clean, noise) in enumerate(validation_loader):
        net.eval()
        with torch.no_grad():
            noisy = noisy.to(device)
            clean = clean.to(device)

            # USING STFT
            if transformation == "stft":
                noisy_abs, noisy_arg = stft_splitter(noisy, window,  n_fft)
                clean_abs, clean_arg = stft_splitter(clean, window, n_fft)
                
                if phase_inc > 0:
                    feat = torch.concat((noisy_abs, noisy_arg[:, phase_feature_indices, :]), dim=1)
                    result, count = net(feat)
                    denoised_abs = result[:, :-n_phase_features,:]
                    denoised_arg = torch.clone(noisy_arg)
                    denoised_arg[:,phase_feature_indices, :] = result[:, n_fft//2+1:, :]
                else:
                    denoised_abs, count = net(noisy_abs)
                    denoised_arg = noisy_arg


                denoised_arg = slayer.axon.delay(denoised_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, n_fft // 4 * out_delay)

                
                idx_abs = torch.where(torch.isnan(denoised_abs))
                idx_arg = torch.where(torch.isnan(denoised_arg))

                clean_rec = stft_mixer(denoised_abs, denoised_arg, window, n_fft)
                
            # USING MFCC with approximated inversion
            elif transformation == "mfcc":
                spectogram_noisy = spec_transformation(noisy)
                spectogram_clean = spec_transformation(clean)

                noisy_abs, noisy_arg = spectogram_noisy.abs(), spectogram_noisy.angle()
                clean_abs, clean_arg = spectogram_clean.abs(), spectogram_clean.angle()

                filter_banked = torch.matmul(H_vals, noisy_abs) + 1e-10
                mfcc = torch.log(filter_banked)
                
                denoised_mfcc, count = net(mfcc)

                noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, n_fft // 4 * out_delay)

                clean_rec, denoised_abs = reconstruct_wave_from_mfcc(n_fft, noisy_abs, noisy_arg, filter_banked, torch.exp(denoised_mfcc), H_vals, map, inv_spec_transformation)
            
            elif transformation == "mdct":
                mdct_features = mdct(noisy)
                denoised_mdct, count = net(mdct_features)
                clean_rec = inv_mdct(denoised_mdct)[:, :480000]


            valid_event_counts.append(count.cpu().data.numpy())
            
            if mse_loss_type == "stft":
                loss = F.mse_loss(denoised_abs, clean_abs)
            elif mse_loss_type == "mfcc":
                mfcc_clean = mse_loss_mfcc_transformation(clean)
                mfcc_denoised = mse_loss_mfcc_transformation(clean_rec)
                loss = F.mse_loss(mfcc_clean, mfcc_denoised)
        
            score = si_snr(clean_rec, clean)

            dnsmos_noisy += np.sum(dnsmos(noisy.cpu().data.numpy()), axis=0)
            dnsmos_clean += np.sum(dnsmos(clean.cpu().data.numpy()), axis=0)
            dnsmos_noise += np.sum(dnsmos(noise.cpu().data.numpy()), axis=0)
            dnsmos_cleaned += np.sum(dnsmos(clean_rec.cpu().data.numpy()), axis=0)

            stats.validation.correct_samples += torch.sum(score).item()
            stats.validation.loss_sum += loss.item()
            stats.validation.num_samples += noisy.shape[0]

            processed = i * validation_loader.batch_size
            total = len(validation_loader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / validation_loader.batch_size
            header_list = [f'Valid: [{processed}/{total} '
                            f'({100.0 * processed / total:.0f}%)]']
            header_list.append(f'Event rate: {[c.item() for c in count]}')
            print(f'\r{header_list[0]}', end='')

    dnsmos_clean /= cur_args.valid_size
    dnsmos_noisy /= cur_args.valid_size
    dnsmos_noise /= cur_args.valid_size
    dnsmos_cleaned /= cur_args.valid_size

    print()
    stats.print(0, i, samples_sec, header=header_list)
    print_stats(stats_file, 'Avg DNSMOS clean   [ovrl, sig, bak]: ' + str(dnsmos_clean))
    print_stats(stats_file, 'Avg DNSMOS noisy   [ovrl, sig, bak]: ' + str(dnsmos_noisy))
    print_stats(stats_file, 'Avg DNSMOS noise   [ovrl, sig, bak]: ' + str(dnsmos_noise))
    print_stats(stats_file, 'Avg DNSMOS cleaned [ovrl, sig, bak]: ' + str(dnsmos_cleaned))

    mean_events = np.mean(valid_event_counts, axis=0)

    neuronops = []
    for block in net.blocks[:-1]:
        neuronops.append(np.prod(block.neuron.shape))

    synops = []
    for events, block in zip(mean_events, net.blocks[1:]):
        synops.append(events * np.prod(block.synapse.shape))
    print_stats(stats_file, f'SynOPS: {synops}')
    print_stats(stats_file, f'Total SynOPS: {sum(synops)} per time-step')
    print_stats(stats_file, f'Total NeuronOPS: {sum(neuronops)} per time-step')
    print_stats(stats_file, f'Time-step per sample: {noisy_abs.shape[-1]}')
    
    # Save validation example audio
    torchaudio.save(eval_out+'valid_example_noisy.wav', noisy[0:1, :].cpu(), 16000)
    torchaudio.save(eval_out+'valid_example_clean.wav', clean[0:1, :].cpu(), 16000)
    torchaudio.save(eval_out+'valid_example_clean_rec.wav', clean_rec[0:1, :].cpu(), 16000)
    
    print_stats(stats_file, "\nLATENCY, QUALITY METRICS AND COMPUTATIONAL METRICS\n")
    
    ###############
    ### LATENCY ###
    ###############
    
    # Buffer latency    
    dt = hop_length / metadata['fs']
    buffer_latency = dt
    print_stats(stats_file, f'Buffer latency: {buffer_latency * 1000} ms')
    
    # ENC / DEC latency
    t_st = datetime.now()
    for i in range(noisy.shape[0]):
        audio = noisy[i].cpu().data.numpy()
        
        # USING STFT
        if transformation == "stft":
            stft = librosa.stft(audio, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
            istft = librosa.istft(stft, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
        # USING MFCC with approximated inversion
        elif transformation == "mfcc":
            spec = spec_transformation(audio)
            abs, arg = spec.abs(), spec.angle()
            filter_banked = torch.matmul(H_vals, abs) + 1e-10
            mfcc = torch.log(filter_banked)
            clean_rec, denoised_abs = reconstruct_wave_from_mfcc(n_fft, abs, arg, filter_banked, torch.exp(mfcc), H_vals, map, inv_spec_transformation)
        elif transformation == "mdct":
            mdct_features = mdct(audio)
            clean_rec = inv_mdct(mdct_features)[:, :480000]

    time_elapsed = (datetime.now() - t_st).total_seconds()

    enc_dec_latency = time_elapsed / noisy.shape[0] / 16000 / 30 * hop_length
    print_stats(stats_file, f'STFT + ISTFT latency: {enc_dec_latency * 1000} ms')


    # N-DNS latency
    dns_delays = []
    max_len = 50000  # Only evaluate for first clip of audio
    for i in range(noisy.shape[0]):
        delay = np.argmax(np.correlate(noisy[i, :max_len].cpu().data.numpy(),
                                    clean_rec[i, :max_len].cpu().data.numpy(),
                                    'full')) - max_len + 1
        dns_delays.append(delay)
    dns_latency = np.mean(dns_delays) / metadata['fs']
    print_stats(stats_file, f'N-DNS latency: {dns_latency * 1000} ms')
    
    #############################
    ### AUDIO QUALITY METRICS ###
    #############################
    
    base_stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                        accuracy_unit='dB')
    nop_stats(validation_loader, base_stats, base_stats.validation, print=False)
    
    si_snr_i = stats.validation.accuracy - base_stats.validation.accuracy
    print_stats(stats_file, f'SI-SNR  (validation set): {stats.validation.accuracy: .2f} dB')
    print_stats(stats_file, f'SI-SNRi (validation set): {si_snr_i: .2f} dB')
    
    #############################
    ### COMPUTATIONAL METRICS ###
    #############################
    
    latency = buffer_latency + enc_dec_latency + dns_latency
    effective_synops_rate = (sum(synops) + 10 * sum(neuronops)) / dt
    synops_delay_product = effective_synops_rate * latency

    print_stats(stats_file, f'Solution Latency                 : {latency * 1000: .3f} ms')
    print_stats(stats_file, f'Power proxy (Effective SynOPS)   : {effective_synops_rate:.3f} ops/s')
    print_stats(stats_file, f'PDP proxy (SynOPS-delay product) : {synops_delay_product: .3f} ops')