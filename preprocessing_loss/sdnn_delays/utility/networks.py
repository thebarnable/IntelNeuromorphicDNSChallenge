import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from time import time

from utility.quantization import *
from utility.helpers import *

from lava.lib.dl import slayer


class Network(torch.nn.Module):
    def __init__(self, threshold=0.1, tau_grad=0.1, scale_grad=0.8, max_delay=64, out_delay=0, n_input=257,
                 n_hidden=512, n_layers=2, batch_norm = False, dropout=False, binarization=False, ternarization=False, quantization=False, obs=None):
        super().__init__()

        self.stft_mean = 0.2
        self.stft_var = 1.5
        self.stft_max = 140
        self.out_delay = out_delay
        self.binarization = binarization
        self.ternarization = ternarization
        self.quantization = quantization
        self.obs = obs

        sigma_params = {  # sigma-delta neuron parameters
            'threshold': threshold,   # delta unit threshold
            'tau_grad': tau_grad,    # delta unit surrogate gradient relaxation parameter
            'scale_grad': scale_grad,  # delta unit surrogate gradient scale parameter
            'requires_grad': False,  # trainable threshold
            'shared_param': True,   # layer wise threshold
        }
        
        if dropout:
            sigma_params["dropout"] = slayer.neuron.dropout.Dropout(1e-4, inplace=True)
        
        sdnn_params = {
            **sigma_params,
            'activation': F.relu,  # activation function
        }

        self.input_quantizer = lambda x: slayer.utils.quantize(x, step=1 / 64)

        if self.quantization:
            print('Network initialization with quantiztion')
            self.blocks = torch.nn.ModuleList([
                slayer.block.sigma_delta.Input(sdnn_params),
                DenseQuant(self.obs, sdnn_params, n_input, n_hidden, weight_norm=False, delay=True, delay_shift=True)] +
                (n_layers-1) * [DenseQuant(self.obs, sdnn_params, n_hidden, n_hidden, weight_norm=False, delay=True, delay_shift=True)] +
                [slayer.block.sigma_delta.Output(sdnn_params, n_hidden, n_input, weight_norm=False)])
        elif self.binarization:
            print('Network initialization with binarization')
            self.blocks = torch.nn.ModuleList([
                slayer.block.sigma_delta.Input(sdnn_params),
                DenseBinary(sdnn_params, n_input, n_hidden, weight_norm=False, delay=True, delay_shift=True)] +
                (n_layers-1) * [DenseBinary(sdnn_params, n_hidden, n_hidden, weight_norm=False, delay=True, delay_shift=True)] +
                [slayer.block.sigma_delta.Output(sdnn_params, n_hidden, n_input, weight_norm=False),])
        elif self.ternarization:
            print('Network initialization with ternarization')
            self.blocks = torch.nn.ModuleList([
                slayer.block.sigma_delta.Input(sdnn_params),
                DenseTernary(sdnn_params, n_input, n_hidden, weight_norm=False, delay=True, delay_shift=True)] +
                (n_layers-1) * [DenseTernary(sdnn_params, n_hidden, n_hidden, weight_norm=False, delay=True, delay_shift=True)] +
                [slayer.block.sigma_delta.Output(sdnn_params, n_hidden, n_input, weight_norm=False)])
        else:
            self.blocks = torch.nn.ModuleList([
                slayer.block.sigma_delta.Input(sdnn_params),
                slayer.block.sigma_delta.Dense(sdnn_params, n_input, n_hidden, weight_norm=False, delay=True, delay_shift=True)] +
                (n_layers-1) * [slayer.block.sigma_delta.Dense(sdnn_params, n_hidden, n_hidden, weight_norm=False, delay=True, delay_shift=True)] +
                [slayer.block.sigma_delta.Output(sdnn_params, n_hidden, n_input, weight_norm=False)])
        
        self.blocks[0].pre_hook_fx = self.input_quantizer
        
        for i in range(1, n_layers+1):
            self.blocks[i].delay.max_delay = max_delay
        
        if batch_norm:
            for i in range(n_layers+1, 1, -1):
                self.blocks.insert(i, slayer.neuron.norm.MeanOnlyBatchNorm(num_features=n_hidden))
                
        print("Created network:")
        print(self)                



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


class ConvNetwork(torch.nn.Module):

    def __init__(self, threshold=0.1, tau_grad=0.1, scale_grad=0.8, max_delay=64, out_delay=0, n_input=257,
                 n_hidden=512, n_layers=5, batch_norm = False, dropout=False, kernel_size=5):
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
        
        if dropout:
            sigma_params["dropout"] = slayer.neuron.dropout.Dropout(1e-4, inplace=True)
        
        sdnn_params = {
            **sigma_params,
            'activation': F.leaky_relu,  # activation function
        }
        
        self.input_quantizer = lambda x: slayer.utils.quantize(x, step=1 / 64)
        
        padding = (0, kernel_size//2)
        kernel_size = (1,kernel_size)

        self.blocks = torch.nn.ModuleList([
            slayer.block.sigma_delta.Input(sdnn_params),
            slayer.block.sigma_delta.Conv(sdnn_params, in_features=n_input, out_features=n_hidden, kernel_size=kernel_size, 
                                          weight_norm=False, delay=False, delay_shift=False, padding=padding)] +
            (n_layers - 1) * [slayer.block.sigma_delta.Conv(sdnn_params, in_features=n_hidden, out_features=n_hidden,
                                                           kernel_size=kernel_size, weight_norm=False, delay=False, delay_shift=False, padding=padding)] +
            [slayer.block.sigma_delta.Output(sdnn_params, n_hidden, n_input, weight_norm=False)]
        )

        self.blocks[0].pre_hook_fx = self.input_quantizer

        for i in range(1, n_layers+1):
            #self.blocks[i].delay.max_delay = max_delay
            #torch.nn.init.normal_(self.blocks[i].synapse.weight)
            pass
            
        #torch.nn.init.normal_(self.blocks[-1].synapse.weight)
            
        if batch_norm:
            for i in range(n_layers+1, 1, -1):
                self.blocks.insert(i, slayer.neuron.norm.MeanOnlyBatchNorm(num_features=n_hidden))
                
                
        print("Created network:")
        print(self)    

    def forward(self, noisy):
        x = noisy - self.stft_mean
        x = torch.unsqueeze(x, dim=2)
        x = torch.unsqueeze(x, dim=4)

        for block in self.blocks:
            x = block(x)
            
        x = torch.squeeze(x)
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

