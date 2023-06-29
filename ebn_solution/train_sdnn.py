# Copyright (C) 2021-22 Intel Corporation
# SPDX-License-Identifier: MIT
# See: https://spdx.org/licenses/

import os
import h5py
import argparse
import numpy as np
from jax import jit, vmap
from jax import numpy as jnp
import jax.example_libraries.optimizers as jopt
import matplotlib.pyplot as plt
from datetime import datetime
import time
import random
import json
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lava.lib.dl import slayer

import jax
jax.config.update('jax_platform_name', 'cpu')

from rockpool.layers.gpl.rate_jax import RecRateEulerJax_IO, H_tanh
from rockpool.layers.training.gpl.jax_trainer import loss_mse_reg
from rockpool import TSContinuous

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from audio_dataloader import DNSAudio
from snr import si_snr

# Helper functions
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

def generate_xor_sample(total_duration, dt, amplitude=1, use_smooth=True, plot=False):
    """
    Generates a temporal XOR signal
    """
    input_duration = 2/3*total_duration
    # Create a time base
    t = np.linspace(0,total_duration, int(total_duration/dt)+1)
    first_duration = np.random.uniform(low=input_duration/10, high=input_duration/4 )
    second_duration = np.random.uniform(low=input_duration/10, high=input_duration/4 )
    end_first = np.random.uniform(low=first_duration, high=2/3*input_duration-second_duration)
    start_first = end_first - first_duration
    start_second = np.random.uniform(low=end_first + 0.1, high=2/3*input_duration-second_duration) # At least 200 ms break
    end_second = start_second+second_duration
    data = np.zeros(int(total_duration/dt)+1)
    i1 = np.random.rand() > 0.5
    i2 = np.random.rand() > 0.5
    response = (((not i1) and i2) or (i1 and (not i2)))
    if(i1):
        a1 = 1
    else:
        a1 = -1
    if(i2):
        a2 = 1
    else:
        a2 = -1
    input_label = 0
    if(a1==1 and a2==1):
        input_label = 0
    elif(a1==1 and a2==-1):
        input_label = 1
    elif(a1==-1 and a2==1):
        input_label = 2
    else:
        input_label = 3
    data[(start_first <= t) & (t < end_first)] = a1
    data[(start_second <= t) & (t < end_second)] = a2
    if(use_smooth):
        sigma = 10
        w = (1/(sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,int(1/dt))-500)**2)/(2*sigma**2))
        w = w / np.sum(w)
        data = amplitude*np.convolve(data, w, "same")
    else:
        data *= amplitude
    target = np.zeros(int(total_duration/dt)+1)
    if(response):
        ar = 1.0
    else:
        ar = -1.0
    target[int(1/dt*(end_second+0.05)):int(1/dt*(end_second))+int(1/dt*0.3)] = ar
    sigma = 20
    w = (1/(sigma*np.sqrt(2*np.pi)))* np.exp(-((np.linspace(1,1000,int(1/dt))-500)**2)/(2*sigma**2))
    w = w / np.sum(w)
    target = np.convolve(target, w, "same")
    target /= np.max(np.abs(target))
    if(plot):
        eps = 0.05
        plt.subplot(211)
        plt.plot(t, data)
        plt.ylim([-amplitude-eps, amplitude+eps])
        plt.subplot(212)
        plt.plot(t, target)
        plt.show()
    return (data[:int(total_duration/dt)], target[:int(total_duration/dt)], input_label)

def save_wav(clean, noisy, denoised, id="", out="audio/", sample=0):
    # clean,noisy,denoised: (batch,samples) or (samples,)
    # out: output directory
    # sample: if batched data, which sample to save

    batched=False
    if len(clean.shape) > 1:
        batched=True
    
    if batched:
        clean=clean[sample,:]
        noisy=noisy[sample,:]
        denoised=denoised[sample,:]

    if not os.path.exists(out):
        os.mkdir(out)

    sf.write(out+"clean"+id+".wav", clean, 16000)
    sf.write(out+"noisy"+id+".wav", noisy, 16000)
    sf.write(out+"denoised"+id+".wav", denoised, 16000)

# Loss functions
## imports
from typing import (
    Dict,
    Tuple,
    Any,
    Callable,
    Union,
    List,
    Optional,
    Collection,
    Iterable,
)
State = Any
Params = Union[Dict, Tuple, List]

## functions
@jit
def loss_mse_reg_default( # Default helper from Rockpool for reservoir nets
    params: Params,                     # Set of packed parameters
    states_t: Dict[str, jnp.ndarray],    # Set of packed state values
    output_batch_t: jnp.ndarray,         # Output rasterised time series [TxO]
    target_batch_t: jnp.ndarray,         # Target rasterised time series [TxO]
    min_tau: float,                     # Minimum time constant
    lambda_mse: float = 1.0,            # Factor when combining loss, on mean-squared error term
    reg_tau: float = 10000.0,           # Factor when combining loss, on minimum time constant limit
    reg_l2_rec: float = 1.0,            # Factor when combining loss, on L2-norm term of recurrent weights
) -> float:
    mse = lambda_mse * jnp.nanmean((output_batch_t - target_batch_t) ** 2)  # output-target MSE loss
    tau_loss = reg_tau * jnp.nanmean(jnp.where(params["tau"] < min_tau, jnp.exp(-(params["tau"] - min_tau)), 0))  # punish tau < min_tau
    w_res_norm = reg_l2_rec * jnp.nanmean(params["w_recurrent"] ** 2)  # punish large w_rec
    return mse + w_res_norm + tau_loss

loss_mse_reg_default_params = {"lambda_mse": 1.0, "reg_tau": 10000.0, "reg_l2_rec": 1.0, "min_tau": 1e-3 * 11}

# Networks
class Baseline(torch.nn.Module):
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

# Training
def train_ebn(args):
    if args.b > 1:
        batch_axis = 0
    else:
        batch_axis = None

    t_all = datetime.now()
    for epoch in range(args.epoch):
        t_st = datetime.now()

        for i, (noisy, clean, noise) in enumerate(train_loader):
            if args.inference:
                break

            # setup data (noisy = clean + noise   [args.b x 480000])
            is_first = (i==0) and (epoch==0)
            if clean.shape[0] != args.b:  # last batch might not be full, Rockpool cant handle that -> skip last batch
                print("Skipping batch %d" % i)
                continue

            if args.b > 1:
                noisy = noisy.unsqueeze(2)
                clean = clean.unsqueeze(2)
            else:
                noisy = noisy.transpose(1,0)
                clean = clean.transpose(1,0)
            noisy = noisy.numpy()
            clean = clean.numpy()

            # perform one optimization iteration (evolve net with inputs + calc loss (avg over batch) + calc gradients (BPTT) + update weights)
            net.reset_time()
            loss, gradients, evolve_fct, denoised = net.train_output_target(noisy,
                                                        clean,
                                                        is_first = is_first,
                                                        batch_axis = batch_axis,
                                                        loss_fcn = loss_mse_reg_default,
                                                        loss_params = loss_mse_reg_default_params,
                                                        optimizer = jopt.adam,
                                                        opt_params = {"step_size": 1e-4})
            stats.training.loss_sum += loss.item() # track running mean over loss

            # evolve net with inputs again to perform inference -> EDIT: recover inference results from train_output_target, hacked into Rockpool
            #denoised, new_state, states_t = evolve_fct() # = rate_jax::RecRateEulerJax::_evolve_functional::evol_func::_get_rec_evolve_jit::rec_evolve_jit
            denoised = np.asarray(denoised.squeeze())
            clean = clean.squeeze()
            noisy = noisy.squeeze()

            # calc si-snr as main accuracy metric
            snr_score = si_snr(denoised, clean)
            stats.training.correct_samples += torch.sum(snr_score).item() # track running mean over snr (hacky: abuse stats.training.accuracy=correct_samples/num_samples=sum(snr)/num_samples)
            stats.training.num_samples += args.b

            # print epoch & timing info to terminal
            processed = i * train_loader.batch_size
            total = len(train_loader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            time_elapsed_total = (datetime.now() - t_all).total_seconds()
            samples_sec = time_elapsed / (i + 1) / train_loader.batch_size
            stats.print(epoch, i, None, [f'Train: {processed}/{total} ({100.0 * processed / total:.0f}%), epoch time: {time_elapsed:.2f}s, total time: {time_elapsed_total:.2f}s'])

            # print loss & snr to tensorboard
            writer.add_scalar('Loss/train', stats.training.loss, i)
            writer.add_scalar('SI-SNR/train', stats.training.accuracy, i)
        
        t_val_st = time.time()
        for i, (noisy, clean, noise) in enumerate(validation_loader):
           
            # setup data (noisy = clean + noise   [args.b x 480000])
            if args.b > 1:
                noisy = noisy.unsqueeze(2)
                clean = clean.unsqueeze(2)
            else:
                noisy = noisy.transpose(1,0)
                clean = clean.transpose(1,0)
            noisy = noisy.numpy()
            clean = clean.numpy()

            # init net if not done yet
            if not net.initialized:
                net.train_output_target(noisy,
                                        clean,
                                        is_first = True,
                                        init_only = True,
                                        batch_axis = batch_axis,
                                        loss_fcn = loss_mse_reg_default,
                                        loss_params = loss_mse_reg_default_params,
                                        optimizer = jopt.adam,
                                        opt_params = {"step_size": 1e-4})
            
            # evolve net with inputs again to perform inference
            denoised, new_state, states_t = net.evolve_directly(noisy) # = rate_jax::RecRateEulerJax::_evolve_functional::evol_func::_get_rec_evolve_jit::rec_evolve_jit

            denoised = np.asarray(denoised.squeeze())
            clean = clean.squeeze()
            noisy = noisy.squeeze()

            # calc si-snr as main accuracy metric
            snr_score = si_snr(denoised, clean)
            stats.validation.correct_samples += torch.sum(snr_score).item() # track running mean over snr (hacky: abuse stats.training.accuracy=correct_samples/num_samples=sum(snr)/num_samples)
            stats.validation.num_samples += args.b

            processed = i * validation_loader.batch_size
            total = len(validation_loader.dataset)
            stats.print(epoch, i, None, [f'Valid: {processed}/{total} ({100.0 * processed / total:.0f}%), time since val start: {time.time() - t_val_st:.2f}s'])

            return

        #writer.add_scalar('Loss/train', stats.training.loss, epoch)
        writer.add_scalar('Loss/valid', stats.validation.loss, epoch)
        #writer.add_scalar('SI-SNR/train', stats.training.accuracy, epoch)
        writer.add_scalar('SI-SNR/valid', stats.validation.accuracy, epoch)

        # stats.update()
        # stats.plot(path=trained_folder + '/')
        if stats.validation.best_accuracy is True:
            net.save_layer("net.json")
        stats.save(trained_folder + '/')

    # net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    # net.export_hdf5(trained_folder + '/network.net')

    # params_dict = {}
    # for key, val in args._get_kwargs():
    #     params_dict[key] = str(val)
    # writer.add_hparams(params_dict, {'SI-SNR': stats.validation.max_accuracy})
    # writer.flush()
    # writer.close()

    net.reset_all()
    net.noise_std = 0.0

def train_baseline(args):
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

        t_st = datetime.now()
        for i, (noisy, clean, noise) in enumerate(validation_loader):
            net.eval()

            with torch.no_grad():
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
        #if stats.validation.best_accuracy is True:
            #torch.save(module.state_dict(), trained_folder + '/network.pt')
        stats.save(trained_folder + '/')

    #net.load_state_dict(torch.load(trained_folder + '/network.pt'))
    # net.export_hdf5(trained_folder + '/network.net')

    # params_dict = {}
    # for key, val in args._get_kwargs():
    #     params_dict[key] = str(val)
    # writer.add_hparams(params_dict, {'SI-SNR': stats.validation.max_accuracy})
    writer.flush()
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-baseline',
                        action='store_true',
                        help='use baseline instead of EBN')
    parser.add_argument('-inference',
                    action='store_true',
                    help='perform only inference on reservoir')
    parser.add_argument('-model',
                        type=str,
                        default="",
                        help='json file containing parameters of reservoir net (saved using net.save_layer())')
    parser.add_argument('-neurons',
                        type=int,
                        default=64,
                        help='neurons in non-spiking teacher reservoir')
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
        np.random.seed(args.seed)
        jax.random.PRNGKey(args.seed)
        random.seed(args.seed)
        #identifier += '_{}{}'.format(args.optim, args.seed)

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
    if args.baseline:
        net = Baseline(args.threshold,
                      args.tau_grad,
                      args.scale_grad,
                      args.dmax,
                      args.out_delay).to(device)
        
        optimizer = torch.optim.RAdam(net.parameters(),
                                lr=args.lr,
                                weight_decay=1e-5)
    else:
        if args.model != "":
            f = open(args.model)
            loaded_net = json.load(f)
            w_in = np.array(loaded_net['w_in'])
            w_rec = np.array(loaded_net['w_recurrent'])
            w_out = np.array(loaded_net['w_out'])
            bias = loaded_net['bias']
            tau = loaded_net['tau']
            name = loaded_net['name']
            if loaded_net['activation_func'] == 'tanh':
                activation_func = H_tanh
            else:
                print("Error: unknown activation function in loaded network")
                exit(1)
            f.close()
        else:
            w_in = 10.0 * (np.random.rand(1, args.neurons) - .5)
            w_rec = 0.2 * (np.random.rand(args.neurons, args.neurons) - .5)
            w_rec -= np.eye(args.neurons) * w_rec
            w_out = 0.4*np.random.uniform(size=(args.neurons, 1))-0.2
            bias = 0.0 * (np.random.rand(args.neurons) - 0.5)
            tau = np.linspace(0.01, 0.1, args.neurons)
            sr = np.max(np.abs(np.linalg.eigvals(w_rec)))
            w_rec = w_rec / sr * 0.95
            activation_func = H_tanh
            name = 'reservoir'

        dt = 1e-3
        duration = 1.0
        net = RecRateEulerJax_IO(activation_func=activation_func,
                                    w_in=w_in,
                                    w_recurrent=w_rec,
                                    w_out=w_out,
                                    tau=tau,
                                    bias=bias,
                                    dt=dt,
                                    noise_std=0.0,
                                    name=name)

    if len(args.gpu) == 1:
        module = net
    else:
        net = torch.nn.DataParallel(net, device_ids=args.gpu)
        module = net.module

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

    stats = slayer.utils.LearningStats(accuracy_str='SI-SNR',
                                       accuracy_unit='dB')

    if args.baseline:
        train_baseline(args)
    else:
        train_ebn(args)
