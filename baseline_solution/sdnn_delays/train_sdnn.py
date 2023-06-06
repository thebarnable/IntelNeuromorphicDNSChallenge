# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os
import h5py
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lava.lib.dl import slayer
from audio_dataloader import DNSAudio
from snr import si_snr
from dnsmos import DNSMOS


def collate_fn(batch):
    noisy, clean, noise = [], [], []

    for sample in batch:
        noisy += [torch.FloatTensor(sample[0])]
        clean += [torch.FloatTensor(sample[1])]
        noise += [torch.FloatTensor(sample[2])]

    return torch.stack(noisy), torch.stack(clean), torch.stack(noise)


def stft_splitter(audio, n_fft=512):
    with torch.no_grad():
        audio_stft = torch.stft(audio,
                                n_fft=n_fft,
                                onesided=True,
                                return_complex=True)
        return audio_stft.abs(), audio_stft.angle()


def stft_mixer(stft_abs, stft_angle, n_fft=512):
    return torch.istft(torch.complex(stft_abs * torch.cos(stft_angle),
                                        stft_abs * torch.sin(stft_angle)),
                        n_fft=n_fft, onesided=True)


class Network(torch.nn.Module):
    def __init__(self, threshold=0.1, tau_grad=0.1, scale_grad=0.8, max_delay=64, out_delay=0):
        super().__init__()
        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay

        sigma_params = { # sigma-delta neuron parameters
            'threshold'     : threshold,   # delta unit threshold
            'tau_grad'      : tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad'    : scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad' : False,  # trainable threshold
            'shared_param'  : True,   # layer wise threshold
        }
        sdnn_params = {
            **sigma_params,
            'activation'    : F.relu, # activation function
        }

        self.input_quantizer = lambda x: slayer.utils.quantize(x, step=1 / 64)

        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Dense(sdnn_params, 257, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Dense(sdnn_params, 512, 512, weight_norm=False, delay=True, delay_shift=True),
            slayer.block.sigma_delta.Output(sdnn_params, 512, 257, weight_norm=False),
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
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

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

    out_delay = args.out_delay
    if len(args.gpu) == 1:
        net = Network(args.threshold,
                      args.tau_grad,
                      args.scale_grad,
                      args.dmax,
                      args.out_delay).to(device)
        module = net
        netEval = InferenceNet(args.threshold,
                      args.tau_grad,
                      args.scale_grad,
                      args.dmax,
                      args.out_delay).to(device)
        moduleEval = netEval
    else:
        net = torch.nn.DataParallel(Network(args.threshold,
                                            args.tau_grad,
                                            args.scale_grad,
                                            args.dmax,
                                            args.out_delay).to(device),
                                    device_ids=args.gpu)
        module = net.module
        netEval = torch.nn.DataParallel(InferenceNet(args.threshold,
                                            args.tau_grad,
                                            args.scale_grad,
                                            args.dmax,
                                            args.out_delay).to(device),
                                    device_ids=args.gpu)
        moduleEval = netEval.module

    # Define optimizer module.
    optimizer = torch.optim.RAdam(net.parameters(),
                                  lr=args.lr,
                                  weight_decay=1e-5)

    train_set = DNSAudio(root=args.path + 'training_set/')
    validation_set = DNSAudio(root=args.path + 'validation_set/')

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

    base_stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                            accuracy_unit='dB')

    # print()
    # print('Base Statistics')
    # nop_stats(train_loader, base_stats, base_stats.training)
    # nop_stats(validation_loader, base_stats, base_stats.validation)
    # print()

    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                       accuracy_unit='dB')

    _, _, _, metadata = train_set[0]
    dnsmos = DNSMOS()

    for epoch in range(args.epoch):
        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(train_loader):
            net.train()
            noisy = noisy.to(device)
            clean = clean.to(device)

            noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft)
            clean_abs, clean_arg = stft_splitter(clean, args.n_fft)

            denoised_abs = net(noisy_abs)
            noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
            clean_abs = slayer.axon.delay(clean_abs, out_delay)
            clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)

            clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft)

            score = si_snr(clean_rec, clean)
            loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))

            assert torch.isnan(loss) == False

            optimizer.zero_grad()
            loss.backward()
            net.validate_gradients()
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            optimizer.step()

            if i < 10:
                net.grad_flow(path=trained_folder + '/')

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
                
                noisy_abs, noisy_arg = stft_splitter(noisy, args.n_fft)
                clean_abs, clean_arg = stft_splitter(clean, args.n_fft)

                denoised_abs, count = netEval(noisy_abs)
                valid_event_counts.append(count.cpu().data.numpy())
                noisy_arg = slayer.axon.delay(noisy_arg, out_delay)
                clean_abs = slayer.axon.delay(clean_abs, out_delay)
                clean = slayer.axon.delay(clean, args.n_fft // 4 * out_delay)
                clean_eval = slayer.axon.delay(clean, args.n_fft * out_delay)

                clean_rec = stft_mixer(denoised_abs, noisy_arg, args.n_fft)
                
                score = si_snr(clean_rec, clean)
                loss = lam * F.mse_loss(denoised_abs, clean_abs) + (100 - torch.mean(score))

                dnsmos_noisy += np.sum(dnsmos(noisy.cpu().data.numpy()), axis=0)
                dnsmos_clean += np.sum(dnsmos(clean_eval.cpu().data.numpy()), axis=0)
                dnsmos_noise += np.sum(dnsmos(noise.cpu().data.numpy()), axis=0)
                dnsmos_cleaned += np.sum(dnsmos(clean_rec.cpu().data.numpy()), axis=0)

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
                header_list.append(f'Event rate: {[c.item() for c in count]}')
                stats.print(epoch, i, samples_sec, header=header_list)

        # Gather DNSMOS, SI-SNR, SynOPs and NeuronOPs statistics
        dnsmos_clean /= len(validation_loader.dataset)
        dnsmos_noisy /= len(validation_loader.dataset)
        dnsmos_noise /= len(validation_loader.dataset)
        dnsmos_cleaned /= len(validation_loader.dataset)

        print()
        print('Avg DNSMOS clean   [ovrl, sig, bak]: ', dnsmos_clean)
        print('Avg DNSMOS noisy   [ovrl, sig, bak]: ', dnsmos_noisy)
        print('Avg DNSMOS noise   [ovrl, sig, bak]: ', dnsmos_noise)
        print('Avg DNSMOS cleaned [ovrl, sig, bak]: ', dnsmos_cleaned)

        mean_events = np.mean(valid_event_counts, axis=0)
        neuronops = []
        for block in net.blocks[:-1]:
            neuronops.append(np.prod(block.neuron.shape))
        synops = []
        for events, block in zip(mean_events, net.blocks[1:]):
            synops.append(events * np.prod(block.synapse.shape))
        print(f'SynOPS: {synops}')
        print(f'Total SynOPS: {sum(synops)} per time-step')
        print(f'Total NeuronOPS: {sum(neuronops)} per time-step')
        print(f'Time-step per sample: {noisy_abs.shape[-1]}')

        ## Latency
        win_length = args.n_fft
        hop_length = args.n_fft // 4
        # buffer latency
        dt = hop_length / metadata['fs']
        buffer_latency = dt
        print(f'Buffer latency: {buffer_latency * 1000} ms')

        # Encode + Decode latency
        t_st = datetime.now()
        for i in range(noisy.shape[0]):
            audio = noisy[i].cpu().data.numpy()
            stft = librosa.stft(audio, n_fft=args.n_fft, win_length=win_length, hop_length=hop_length)
            istft = librosa.istft(stft, n_fft=args.n_fft, win_length=win_length, hop_length=hop_length)

        time_elapsed = (datetime.now() - t_st).total_seconds()

        enc_dec_latency = time_elapsed / noisy.shape[0] / 16000 / 30 * hop_length
        print(f'STFT + ISTFT latency: {enc_dec_latency * 1000} ms')

        # N-DNS latency
        dns_delays = []
        max_len = 50000  # Only evaluate for first clip of audio
        for i in range(noisy.shape[0]):
            delay = np.argmax(np.correlate(noisy[i, :max_len].cpu().data.numpy(),
                                        clean_rec[i, :max_len].cpu().data.numpy(),
                                        'full')) - max_len + 1
            dns_delays.append(delay)
        dns_latency = np.mean(dns_delays) / metadata['fs']
        print(f'N-DNS latency: {dns_latency * 1000} ms')

        # Audio Quality
        base_stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                                accuracy_unit='dB')
        nop_stats(validation_loader, base_stats, base_stats.validation, print=False)

        si_snr_i = stats.validation.accuracy - base_stats.validation.accuracy
        print(f'SI-SNR  (validation set): {stats.validation.accuracy: .2f} dB')
        print(f'SI-SNRi (validation set): {si_snr_i: .2f} dB')

        # Computational Metrics
        latency = buffer_latency + enc_dec_latency + dns_latency
        effective_synops_rate = (sum(synops) + 10 * sum(neuronops)) / dt
        synops_delay_product = effective_synops_rate * latency

        print(f'Solution Latency                 : {latency * 1000: .3f} ms')
        print(f'Power proxy (Effective SynOPS)   : {effective_synops_rate:.3f} ops/s')
        print(f'PDP proxy (SynOPS-delay product) : {synops_delay_product: .3f} ops')

        writer.add_scalar('Loss/train', stats.training.loss, epoch)
        writer.add_scalar('Loss/valid', stats.validation.loss, epoch)
        writer.add_scalar('SI-SNR/train', stats.training.accuracy, epoch)
        writer.add_scalar('SI-SNR/valid', stats.validation.accuracy, epoch)
   
        dnsmos_stats_names = ['ovrl', 'sig', 'bak']
        dnsmos_clean_writer = {dnsmos_stats_names[i]: dnsmos_clean[i] for i in range(len(dnsmos_stats_names))}
        writer.add_scalars('Avg DNSMOS clean', dnsmos_clean_writer, epoch)
        dnsmos_noisy_writer = {dnsmos_stats_names[i]: dnsmos_noisy[i] for i in range(len(dnsmos_stats_names))}
        writer.add_scalars('Avg DNSMOS clean', dnsmos_noisy_writer, epoch)
        dnsmos_noise_writer = {dnsmos_stats_names[i]: dnsmos_noise[i] for i in range(len(dnsmos_stats_names))}
        writer.add_scalars('Avg DNSMOS clean', dnsmos_noise_writer, epoch)
        dnsmos_cleaned_writer = {dnsmos_stats_names[i]: dnsmos_cleaned[i] for i in range(len(dnsmos_stats_names))}
        writer.add_scalars('Avg DNSMOS clean', dnsmos_cleaned_writer, epoch)
        
        synops_writer = {['_', '_', '_'][i]: dnsmos_cleaned[i] for i in range(len(['_', '_', '_']))}
        writer.add_scalars('SynOPS', synops_writer, epoch)
        writer.add_scalar('Total SynOPS (per time-step)', sum(synops), epoch)
        writer.add_scalar('Total NeuronOPS (per time-step)', sum(neuronops), epoch)
        writer.add_scalar('Time-step per sample', noisy_abs.shape[-1], epoch)

        writer.add_scalar('Buffer latency (ms)', buffer_latency * 1000, epoch)
        writer.add_scalar('STFT + ISTFT latency (ms)', enc_dec_latency * 1000, epoch)
        writer.add_scalar('N-DNS latency (ms)', dns_latency * 1000, epoch)

        writer.add_scalar('SI-SNR (dB)', stats.validation.accuracy, epoch)
        writer.add_scalar('SI-SNRi (dB)', si_snr_i, epoch)

        writer.add_scalar('solution latency (ms)', latency, epoch)
        writer.add_scalar('power proxy (Effective SynOPS; ops/s)', effective_synops_rate, epoch)
        writer.add_scalar('pdp proxy (SynOPS-delay product; ops) ', synops_delay_product, epoch)

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