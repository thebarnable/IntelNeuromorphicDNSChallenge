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
# import torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.modules.utils import _single
from typing import Optional

from lava.lib.dl import slayer
from audio_dataloader import DNSAudio
from snr import si_snr

from datetime import datetime
#ct_str = datetime.now().strftime("%Y%m%d%H%M%S")
ct_str = datetime.now().strftime("%Y%m%d")

import pdb
import itertools

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

class Baseline(torch.nn.Module):
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

class Conv1d_binary(torch.nn.Conv1d):
    # overwrite Dense layer forward method
    def forward(self, input):
        # binarize weights
        weight_bin = self.weight.clone()
        weight_bin = weight_bin.add_(1).div_(2).clamp_(0,1)
        weight_bin.round()
        weight_bin.mul_(2).add_(-1)
        if self._pre_hook_fx is None:
            weight = weight_bin
        else:
            weight = self._pre_hook_fx(weight_bin)

        if len(input.shape) == 3:
            old_shape = input.shape
            return F.conv3d(  # bias does not need pre_hook_fx. Its disabled
                input.reshape(old_shape[0], -1, 1, 1, old_shape[-1]),
                weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            ).reshape(old_shape[0], -1, old_shape[-1])
        else:
            return F.conv3d(
                input, weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            )
        
    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        weight_bin = weight.clone()
        weight_bin = weight_bin.add_(1).div_(2).clamp_(0,1)
        weight_bin.round()
        weight_bin.mul_(2).add_(-1)
        weight = weight_bin

        if self.padding_mode != 'zeros':
            return F.conv1d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _single(0), self.dilation, self.groups)
        return F.conv1d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward(input, self.weight, self.bias)

class Network(torch.nn.Module):
    def __init__(self,inp,k,c,d,  threshold=0.1, tau_grad=0.1, scale_grad=0.8, max_delay=64, out_delay=0):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay

        #unet like architecture
        # self.blocks = torch.nn.ModuleList([
        #     torch.nn.Conv1d(257, c, kernel_size=k, padding='same', bias=False),
        #     torch.nn.BatchNorm1d(c),
        #     torch.nn.ReLU(inplace=True),
        #     #torch.nn.MaxPool1d(2)
        #     torch.nn.Conv1d(c, c, kernel_size=k, padding='same', bias=False),
        #     torch.nn.BatchNorm1d(c),
        #     torch.nn.ReLU(inplace=True),
        #     #torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     torch.nn.Conv1d(c, 257, kernel_size=k, padding='same', bias=False),
        #     torch.nn.BatchNorm1d(257),
        #     torch.nn.ReLU(inplace=True),
        # ])
        self.blocks = torch.nn.ModuleList([
            Conv1d_binary(inp, c, kernel_size=k, padding='same', bias=False),
            torch.nn.BatchNorm1d(c),
            torch.nn.ReLU(inplace=True),
        ])

        for ii in range(d):
            self.blocks.append(Conv1d_binary(c, c, kernel_size=k, padding='same', bias=False))
            self.blocks.append(torch.nn.BatchNorm1d(c))
            self.blocks.append(torch.nn.ReLU(inplace=True))

        self.blocks.append(Conv1d_binary(c, inp, kernel_size=k, padding='same', bias=False))
        self.blocks.append(torch.nn.BatchNorm1d(inp))
        self.blocks.append(torch.nn.ReLU(inplace=True))

    def forward(self, noisy):
        # pdb.set_trace()
        x = noisy
        for block in self.blocks:
            x = block(x)

        return x

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
                        default=64,
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
    parser.add_argument('-id',
                        type=str,
                        default='',
                        help='experiment differentiater string')
    parser.add_argument('-out',
                        type=str,
                        default='runs/',
                        help='results path')    
    parser.add_argument('-opt',
                        type=str,
                        default='adam',
                        help='optimizer function (as defined in jax::optimizers.py)')    
    parser.add_argument('-seed',
                        type=int,
                        default=0,
                        help='random seed of the experiment')
    parser.add_argument('-epoch',
                        type=int,
                        default=200,
                        help='number of epochs to run')
    parser.add_argument('-path',
                        type=str,
                        default='/mnt/data4tb/stadtmann/dns_challenge_4/datasets_fullband/',
                        help='dataset path')
    parser.add_argument('-baseline',
                        action='store_true',
                        help='use baseline SNN instead of CNN')
    parser.add_argument('-cpu',
                        action='store_true',
                        help='force CPU usage even if GPU is available')
    
    parser.add_argument("-n_mfcc", type=int, default=50, help="Number of MFCC components (only considered if transformation is mfcc or mse_loss_type is mfcc)")
    parser.add_argument("-stft_window", type=str, choices=["hann", "hamming", "gaussian", "bartlett", "rectangle"],
                        default="rectangle", help="window function for STFT (considered if transformation=stft or transformation=mfcc)")
    parser.add_argument("-transformation", type=str, default="stft", choices=["stft", "mfcc"], help="transformation that is applied as encoding/decoding")
    parser.add_argument("-mse_loss_type", type=str, default = "stft", choices = ["stft", "mfcc"], help = "Which coefficients to use for MSE part of the loss function")

    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # identifier += '_{}{}'.format(args.optim, args.seed)

    # setup results folders in args.out + args.id
    if args.id == "":
        args.id = datetime.today().strftime('exp_%Y%m%d_%H%M%S')
    trained_folder = os.path.abspath(args.out + "/" + args.id)
    print(F"Starting experiment {args.id}. Results in {trained_folder}.")
    
    writer = SummaryWriter(trained_folder)
    os.makedirs(trained_folder, exist_ok=True)

    with open(trained_folder + '/args.txt', 'wt') as f:
        for arg, value in sorted(vars(args).items()):
            f.write('{} : {}\n'.format(arg, value))

    # set misc parameters
    kk = 5
    cc = 256
    dd = 5

    lam = args.lam

    if torch.cuda.device_count() > 0 and not args.cpu:
        print('Using GPUs {}'.format(args.gpu))
        device = torch.device('cuda:{}'.format(args.gpu[0]))
    else:
        print('No GPU found. Defaulting to CPU')
        device = torch.device('cpu')

    out_delay = args.out_delay

    # set MFCC and/or STFT lambdas and params
    window_fn = None  # treated as rectangular window function
    window = None
    if args.stft_window != 'rectangle':
        window_fn = lambda n : getattr(torch, args.stft_window+'_window')(n, device=device)
        window = window_fn(args.n_fft)
    # if args.stft_window == "hann":
    #     window_fn = lambda n : torch.hann_window(n, device=device)
    # elif args.stft_window == "hamming":
    #     window_fn = lambda n : torch.hamming_window(n, device=device)
    # elif args.stft_window == "bartlett":
    #     window_fn = lambda n : torch.bartlett_window(n, device=device)
    
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

    if args.mse_loss_type == "mfcc":
        loss_transf = torchaudio.transforms.MFCC(n_mfcc = args.n_mfcc, 
                                                melkwargs = {"window_fn" : lambda n:torch.hann_window(n, device = device), 
                                                            "n_fft" : 512, 
                                                            "hop_length" : 128}).to(device)
    else:
        loss_transf = lambda n: n

    n_input = args.n_fft // 2 +1 if args.transformation == "stft" else args.n_mfcc
    #n_hidden = args.n_fft if args.transformation == "stft" else int(args.n_mfcc * args.hidden_input_ratio)
    if args.baseline:
        n_hidden = args.n_fft if args.transformation == "stft" else int(args.n_mfcc * 2)
        if len(args.gpu) == 1:
            net = Baseline(args.threshold,
                        args.tau_grad,
                        args.scale_grad,
                        args.dmax,
                        args.out_delay,
                        n_input, n_hidden).to(device)
            module = net
        else:
            net = torch.nn.DataParallel(Baseline(args.threshold,
                                                args.tau_grad,
                                                args.scale_grad,
                                                args.dmax,
                                                args.out_delay,
                                                n_input, n_hidden).to(device),
                                        device_ids=args.gpu)
            module = net.module
    else:
        if len(args.gpu) == 1:
            net = Network(n_input,kk,cc,dd,
                        args.threshold,
                        args.tau_grad,
                        args.scale_grad,
                        args.dmax,
                        args.out_delay).to(device)
            module = net
        else:
            net = torch.nn.DataParallel(Network(n_input, kk,cc,dd,
                                                args.threshold,
                                                args.tau_grad,
                                                args.scale_grad,
                                                args.dmax,
                                                args.out_delay).to(device),
                                        device_ids=args.gpu)
            module = net.module

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150], gamma=0.1)

    train_set = DNSAudio(root=args.path + 'training_set/')
    validation_set = DNSAudio(root=args.path + 'validation_set/')

    train_loader = DataLoader(train_set,
                            batch_size=args.b,
                            shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=1,
                            pin_memory=True)
    validation_loader = DataLoader(validation_set,
                                batch_size=args.b,
                                shuffle=True,
                                collate_fn=collate_fn,
                                num_workers=1,
                                pin_memory=True)

    base_stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                            accuracy_unit='dB')
    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                    accuracy_unit='dB')

    print("Starting training")
    t_all = datetime.now()
    for epoch in range(args.epoch):
        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(train_loader):
            net.train()
            noisy = noisy.to(device)
            clean = clean.to(device)

            if args.transformation == 'stft':
                noisy_abs, noisy_arg = stft_splitter(noisy, window, args.n_fft)
                clean_abs, clean_arg = stft_splitter(clean, window, args.n_fft)

                denoised_abs = net(noisy_abs)
                #pdb.set_trace()
                noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

                clean_rec = stft_mixer(denoised_abs, noisy_arg, window, args.n_fft)
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


            score = si_snr(clean_rec, clean)
            loss = lam * F.mse_loss(loss_transf(denoised_abs), loss_transf(clean_abs)) + (100 - torch.mean(score))

            assert torch.isnan(loss) == False

            optimizer.zero_grad()
            loss.backward()
            net.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            # if i < 10:
            #     net.grad_flow(path=trained_folder + '/')

            if torch.isnan(score).any():
                score[torch.isnan(score)] = 0

            stats.training.correct_samples += torch.sum(score).item()
            stats.training.loss_sum += loss.item()
            stats.training.num_samples += noisy.shape[0]

            processed = i * train_loader.batch_size
            total = len(train_loader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            time_elapsed_total = (datetime.now() - t_all).total_seconds()
            samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
            stats.print(epoch, i, None, [f'Train: {processed}/{total} ({100.0 * processed / total:.0f}%), epoch time: {time_elapsed:.2f}s, total time: {time_elapsed_total:.2f}s'])

        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(validation_loader):
            net.eval()

            with torch.no_grad():
                noisy = noisy.to(device)
                clean = clean.to(device)
                
                noisy_abs, noisy_arg = stft_splitter(noisy, window, args.n_fft)
                clean_abs, clean_arg = stft_splitter(clean, window, args.n_fft)

                denoised_abs = net(noisy_abs)
                noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

                clean_rec = stft_mixer(denoised_abs, noisy_arg, window, args.n_fft)
                
                score = si_snr(clean_rec, clean)
                loss = lam * F.mse_loss(loss_transf(denoised_abs), loss_transf(clean_abs)) + (100 - torch.mean(score))
                stats.validation.correct_samples += torch.sum(score).item()
                stats.validation.loss_sum += loss.item()
                stats.validation.num_samples += noisy.shape[0]

                processed = i * validation_loader.batch_size
                total = len(validation_loader.dataset)
                time_elapsed = (datetime.now() - t_st).total_seconds()
                samples_sec = time_elapsed / (i + 1) / validation_loader.batch_size
                stats.print(epoch, i, None, [f'Valid: {processed}/{total} ({100.0 * processed / total:.0f}%), time since val start: {time_elapsed:.2f}s'])

        writer.add_scalar('Loss/train', stats.training.loss, epoch)
        writer.add_scalar('Loss/valid', stats.validation.loss, epoch)
        writer.add_scalar('SI-SNR/train', stats.training.accuracy, epoch)
        writer.add_scalar('SI-SNR/valid', stats.validation.accuracy, epoch)

        print(f'Epoch: {epoch}, loss: {stats.validation.loss}, si-snr: {stats.validation.accuracy}')

        stats.update()
        stats.plot(path=trained_folder + '/')
        if stats.validation.best_accuracy is True:
            torch.save(module.state_dict(), trained_folder + '/network.pt')
        stats.save(trained_folder + '/')

        scheduler.step()

    net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    #net.export_hdf5(trained_folder + '/network.net')

    params_dict = {}
    for key, val in args._get_kwargs():
        params_dict[key] = str(val)
    #writer.add_hparams(params_dict, {'SI-SNR': stats.validation.max_accuracy})
    writer.flush()
    writer.close()
