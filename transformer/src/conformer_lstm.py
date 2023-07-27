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
  def __init__(self, sample_rate, n_fft, frame, stride, device, batch, phase, network, length=3001, no_gpus=1):
    print("CONFORMER_LSTM")
    super(DNSModelConformer, self).__init__()
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
    self.depthwise_conv_kernel_size = network["depthwise_conv_kernel_size"]
    self.lstm_layers = 1
    if "lstm_layers" in network:
      self.lstm_layers = network["lstm_layers"]

    self.length = length - 1
    
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
    self.lstm = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=self.lstm_layers)
    self.linear_out = nn.Linear(self.d_model, self.fft_out_size)


  def forward(self, src, lengths):

    src_spec = src

    ## Source ##
    #print("src_spec: ", src_spec.shape)
    src_spec = src_spec.permute(0,2,1)

    #print("src_spec: ", src_spec.shape)
    reshaped_src = src_spec
    #reshaped_src = src_spec.view(self.batch * src_spec.shape[1], self.fft_out_size)
    emb_src = self.swish(self.linear_in(reshaped_src))
    #print("emb_src: ", emb_src.shape)
    #emb_src = emb_src.view(self.batch, src_spec.shape[1], self.d_model)
    #print("emb_src: ", emb_src.shape)
    emb_src = self.positional_encoder(emb_src)
      
    #print(emb_src.shape)
    transformer_out = self.conformer(emb_src, lengths)[0]

    #print("transformer_out ", transformer_out.shape)
    lstm_out, _ = self.lstm(transformer_out)
    #print("lstm_out ", lstm_out.shape)
    #reshaped_out = lstm_out.view(self.batch * src_spec.shape[1], self.d_model)
    #print("reshaped_out ", reshaped_out.shape)
    lin_out = self.swish(self.linear_out(lstm_out))
    #print("lin_out ", lin_out.shape)
    #lin_out = lin_out.view(self.batch, src_spec.shape[1], self.fft_out_size)
    #print("lin_out ", lin_out.shape)
    lin_out = lin_out.permute(0,2,1)
    #print("lin_out ", lin_out.shape, flush=True)


    return lin_out[:,:,:]

