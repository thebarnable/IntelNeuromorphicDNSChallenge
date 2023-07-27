import torch
import torch.nn.functional as F
from lava.lib.dl import slayer



class LayerDenseBinary(slayer.synapse.layer.Dense):
    # overwrite Dense layer forward method
    def forward(self, input):
        
        # binarize weights
        self.binarize()

        if len(input.shape) == 3:
            old_shape = input.shape
            return F.conv3d(  # bias does not need pre_hook_fx. Its disabled
                input.reshape(old_shape[0], -1, 1, 1, old_shape[-1]),
                self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            ).reshape(old_shape[0], -1, old_shape[-1])
        else:
            return F.conv3d(
                input, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            )
            
    def binarize(self):
        self.save_non_binary_weights()
        weight_bin = self.weight.clone().detach()
        weight_bin.add_(1).div_(2).clamp_(0,1)
        weight_bin.round_()
        weight_bin = weight_bin.mul_(2).add_(-1)
        
        if self._pre_hook_fx is not None:
            weight_bin = self._pre_hook_fx(weight_bin)
        
        self.weight.data.copy_(weight_bin.detach())
        
    def save_non_binary_weights(self):
        if not hasattr(self, "non_bin_weight"):
            self.non_bin_weight = self.weight.clone().detach()
        self.non_bin_weight.data.copy_(self.weight.detach())
        
    def load_non_binary_weights(self):
        self.weight.data.copy_(self.non_bin_weight.detach())
        
    def clamp(self):
        self.weight.data.clamp_(-1,1).detach_()
        
class LayerDenseTernary(slayer.synapse.layer.Dense):
    # overwrite Dense layer forward method
    def forward(self, input):
        
        # ternarize weights
        self.ternarize()

        if len(input.shape) == 3:
            old_shape = input.shape
            return F.conv3d(  # bias does not need pre_hook_fx. Its disabled
                input.reshape(old_shape[0], -1, 1, 1, old_shape[-1]),
                self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            ).reshape(old_shape[0], -1, old_shape[-1])
        else:
            return F.conv3d(
                input, self.weight, self.bias,
                self.stride, self.padding, self.dilation, self.groups,
            )
            
    def ternarize(self):
        self.save_non_ternary_weights()
        
        weight_ter = self.weight.clone().detach()
        thresh = weight_ter.abs().sum() / weight_ter.data.numel()
        thresh *= 0.7

        weight_ter[weight_ter>thresh] = 1
        weight_ter[weight_ter<-thresh] = -1
        weight_ter[weight_ter.data.abs()<=thresh] = 0

        if self._pre_hook_fx is not None:
            weight_ter = self._pre_hook_fx(weight_ter)
        
        self.weight.data.copy_(weight_ter.detach())
        
    def save_non_ternary_weights(self):
        if not hasattr(self, "non_tern_weight"):
            self.non_tern_weight = self.weight.clone().detach()
        self.non_tern_weight.data.copy_(self.weight.detach())
        
    def load_non_ternary_weights(self):
        self.weight.data.copy_(self.non_tern_weight.detach())
        
    def clamp(self):
        self.weight.data.clamp_(-1,1).detach_()

class LayerDenseQuant(slayer.synapse.layer.Dense):
    def __init__(self, obs,
        in_neurons, out_neurons,
        weight_scale=1, weight_norm=False, pre_hook_fx=None):
        super(LayerDenseQuant, self).__init__(in_neurons, out_neurons, 
                weight_scale, weight_norm, pre_hook_fx)
        self.obs = obs
        
    # overwrite Dense layer forward method
    def forward(self, input):
        # ternarize weights
        weight_quant = self.weight.clone()
        _ = self.obs(weight_quant)
        scale, zero_point = self.obs.calculate_qparams()
        scale = scale.cuda().type_as(weight_quant)
        zero_point = zero_point.cuda().type_as(weight_quant)
        weight_quant = torch.quantize_per_tensor(weight_quant, scale, zero_point)

        if self._pre_hook_fx is None:
            weight = weight_quant
        else:
            weight = self._pre_hook_fx(weight_quant)

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
        
class DenseBinary(slayer.block.sigma_delta.AbstractSDRelu, slayer.block.base.AbstractDense):
    # overwrite init of original slayer.block.sigma_delta.Dense to use Dense with modified forward method
    def __init__(self, *args, **kwargs):
        super(DenseBinary, self).__init__(*args, **kwargs)
        self.synapse = LayerDenseBinary(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params

class DenseTernary(slayer.block.sigma_delta.AbstractSDRelu, slayer.block.base.AbstractDense):
    # overwrite init of original slayer.block.sigma_delta.Dense to use Dense with modified forward method
    def __init__(self, *args, **kwargs):
        super(DenseTernary, self).__init__(*args, **kwargs)
        self.synapse = LayerDenseTernary(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params

class DenseQuant(slayer.block.sigma_delta.AbstractSDRelu, slayer.block.base.AbstractDense):
    # overwrite init of original slayer.block.sigma_delta.Dense to use Dense with modified forward method
    def __init__(self, obs, *args, **kwargs):
        super(DenseQuant, self).__init__(*args, **kwargs)
        self.synapse = LayerDenseQuant(obs, **self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params
   
class LayerTernaryConv(slayer.synapse.layer.Conv):
    def __init__(
        self, obs, in_features, out_features, kernel_size,
        stride=1, padding=0, dilation=1, groups=1,
        weight_scale=1, weight_norm=False, pre_hook_fx=None
    ):
        super(LayerTernaryConv, self).__init__(in_features, out_features, kernel_size,
        stride, padding, dilation, groups,
        weight_scale, weight_norm, pre_hook_fx)
        
        self.obs = obs
    
    def forward(self, input):
        self.ternarize()
        
        return super(LayerTernaryConv, self).forward(input)
            
    def ternarize(self):
        self.save_non_ternary_weights()
        
        weight_ter = self.weight.clone().detach()
        thresh = weight_ter.abs().sum() / weight_ter.data.numel()
        thresh *= 0.7

        weight_ter[weight_ter>thresh] = 1
        weight_ter[weight_ter<-thresh] = -1
        weight_ter[weight_ter.data.abs()<=thresh] = 0

        if self._pre_hook_fx is not None:
            weight_ter = self._pre_hook_fx(weight_ter)
        
        self.weight.data.copy_(weight_ter.detach())
        
    def save_non_ternary_weights(self):
        if not hasattr(self, "non_tern_weight"):
            self.non_tern_weight = self.weight.clone().detach()
        self.non_tern_weight.data.copy_(self.weight.detach())
        
    def load_non_ternary_weights(self):
        self.weight.data.copy_(self.non_tern_weight.detach())
        
    def clamp(self):
        self.weight.data.clamp_(-1,1).detach_()
        
class TernaryConv(slayer.block.sigma_delta.AbstractSDRelu, slayer.block.base.AbstractConv):
    def __init__(self, obs, *args, **kwargs):
        super(TernaryConv, self).__init__(*args, **kwargs)
        self.synapse = LayerTernaryConv(obs, **self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params