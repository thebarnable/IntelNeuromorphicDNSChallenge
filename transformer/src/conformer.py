import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
import math

# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
class PositionalEncoding(nn.Module):
  def __init__(self, dim_model, dropout_p, max_len):
    super().__init__()
    # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    # max_len determines how far the position can have an effect on a token (window)
    
    # Info
    self.dropout = nn.Dropout(dropout_p)
    
    # Encoding - From formula
    pos_encoding = torch.zeros(max_len, dim_model)
    positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
    division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
    
    # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
    pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
    
    # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
    pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
    
    # Saving buffer (same as parameter without gradients needed)
    pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
    self.register_buffer("pos_encoding",pos_encoding)
    
  def forward(self, token_embedding: torch.tensor) -> torch.tensor:
    # Residual connection + pos encoding
    return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

    

class DNSModelConformer(nn.Module):
  def __init__(self, sample_rate, n_fft, frame, stride, device, batch, phase, network, no_gpus=1):
    super(DNSModelConformer, self).__init__()
    print("CONFORMER")
    self.model_type = "DNSModelBasic"
    self.sample_rate = sample_rate
    self.n_fft = n_fft
    self.frame = frame
    self.stride = stride

    self.d_model = network["d_model"]
    self.heads = network["heads"]
    self.num_layers = network["num_layers"]
    self.d_ff = network["d_ff"]
    self.dropout = network["dropout"]
    self.num_layers = network["num_layers"]
    self.depthwise_conv_kernel_size = network["depthwise_conv_kernel_size"]

    self.length = 3001 - 1
    
    self.batch = int(batch/no_gpus)
    self.device = device

    if phase: # if phase is considered, the dimensions for magnitude and phase are concatenated
      self.fft_out_size = (int(self.n_fft/2) + 1) * 2
    else:
      self.fft_out_size = int(self.n_fft/2) + 1

    self.positional_encoder = PositionalEncoding(
        dim_model=self.d_model, dropout_p=self.dropout, max_len=self.length
    )
    self.linear_in = nn.Linear(self.fft_out_size, self.d_model)
    self.swish = nn.SiLU()
    self.conformer = torchaudio.models.Conformer(
        input_dim=self.d_model,
        num_heads=self.heads,
        ffn_dim=self.d_ff,
        num_layers=self.num_layers,
        depthwise_conv_kernel_size=self.depthwise_conv_kernel_size,
        dropout=self.dropout
    )
    self.linear_out = nn.Linear(self.d_model, self.fft_out_size)


  def forward(self, src, lengths):

    src_spec = src

    ## Source ##
    #print("src_spec: ", src_spec.shape)
    src_spec = src_spec.permute(0,2,1)
    #flattened_src = src_spec

    #print("flattened_src: ", flattened_src.shape)
    #print(self.fft_out_size, flush=True)
    #reshaped_src = flattened_src.reshape(-1, self.fft_out_size)
    emb_src = self.swish(self.linear_in(src_spec))
    #print("emb_src: ", emb_src.shape)
    #emb_src = emb_src.view(self.batch, src_spec.shape[1], self.d_model)
    #print("emb_src: ", emb_src.shape)
    emb_src = self.positional_encoder(emb_src)
      
    #print(emb_src.shape)
    transformer_out = self.conformer(emb_src, lengths)[0]

    #print("transformer_out ", transformer_out.shape)
    #reshaped_trans0 = transformer_out.reshape(-1, self.d_model)
    #print("reshaped_trans0 ", reshaped_trans0.shape)
    lin_out = self.swish(self.linear_out(transformer_out))
    #lin_out = lin_out.reshape(self.batch, src_spec.shape[1], self.fft_out_size)
    #print("lin_out ", lin_out.shape)
    lin_out = lin_out.permute(1,2,0)
    #print("lin_out ", lin_out.shape)


    return lin_out[:,:,:-1]

