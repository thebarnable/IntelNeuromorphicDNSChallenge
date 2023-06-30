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

    

class DNSModel(nn.Module):
  def __init__(self, sample_rate, n_fft, frame, stride, device, batch, phase, network):
    super(DNSModel, self).__init__()
    self.model_type = "DNSModelBasic"
    self.sample_rate = sample_rate
    self.n_fft = n_fft
    self.frame = frame
    self.stride = stride

    self.d_model = network["d_model"]
    self.heads = network["heads"]
    self.enc = network["enc"]
    self.dec = network["dec"]
    self.d_ff = network["d_ff"]
    self.dropout = network["dropout"]

    self.length = 3001 - 1
    
    self.batch = batch
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
    self.transformer = nn.Transformer(
        d_model=self.d_model,
        nhead=self.heads,
        num_encoder_layers=self.enc,
        num_decoder_layers=self.dec,
        dim_feedforward=self.d_ff,
        dropout=self.dropout
    )
    self.linear_out = nn.Linear(self.d_model, self.fft_out_size)


  def forward(self, src, tgt, TRAIN=True):

    src_spec = None
    tgt_spec = None
    if TRAIN:
      # shift target by one
      src_spec = src
      #print(spec_src.shape)
      tgt_spec = tgt[:,:,:-1].contiguous()
    else:
      src_spec = src
      tgt_spec = tgt

    ## Source ##
    #print("src_spec: ", src_spec.shape)
    src_spec = src_spec.permute(0,2,1)
    flattened_src = src_spec

    #print("flattened_src: ", flattened_src.shape)
    #print(self.fft_out_size, flush=True)
    reshaped_src = flattened_src.reshape(-1, self.fft_out_size)
    emb_src = self.swish(self.linear_in(reshaped_src))
    #print("emb_src: ", emb_src.shape)
    emb_src = emb_src.view(self.batch, src_spec.shape[1], self.d_model)
    #print("emb_src: ", emb_src.shape)
    emb_src = self.positional_encoder(emb_src)
    emb_src = emb_src.permute(1,0,2)
    #print("emb_src: ", emb_src.shape)

    ## Target ##
    tgt_mask = self.get_tgt_mask(tgt_spec.shape[2]).to(self.device)
    tgt_spec = tgt_spec.permute(0,2,1)
    flattened_tgt = tgt_spec
    #print("flattened_src: ", flattened_src.shape)
    reshaped_tgt = flattened_tgt.reshape(-1, self.fft_out_size)
    emb_tgt = self.swish(self.linear_in(reshaped_tgt))
    #print("emb_tgt: ", emb_tgt.shape)
    emb_tgt = emb_tgt.view(self.batch, tgt_spec.shape[1], self.d_model)
    #print("emb_tgt: ", emb_tgt.shape)
    emb_tgt = self.positional_encoder(emb_tgt)
    emb_tgt = emb_tgt.permute(1,0,2)
    #print("emb_tgt: ", emb_tgt.shape)
      
    #print(emb_src.shape)
    #print(emb_tgt.shape)
    transformer_out = self.transformer(emb_src, emb_tgt, tgt_mask=tgt_mask)

    reshaped_trans0 = transformer_out.view(-1, self.d_model)
    #print("reshaped_trans0 ", reshaped_trans0.shape)
    lin_out = self.linear_out(reshaped_trans0)
    #print("lin_out ", lin_out.shape)
    lin_out = lin_out.view(tgt_spec.shape[1], self.batch, self.fft_out_size)
    lin_out = lin_out.permute(1,2,0)
    #print("lin_out ", lin_out.shape)


    return lin_out

    

  def get_tgt_mask(self, size) -> torch.tensor:
    # Generates a squeare matrix where the each row allows one word more to be seen
    mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
    mask = mask.float()
    mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
    mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
    
    # EX for size=5:
    # [[0., -inf, -inf, -inf, -inf],
    #  [0.,   0., -inf, -inf, -inf],
    #  [0.,   0.,   0., -inf, -inf],
    #  [0.,   0.,   0.,   0., -inf],
    #  [0.,   0.,   0.,   0.,   0.]]
    
    return mask
    

