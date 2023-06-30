import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import wavfile
import numpy as np
from datetime import datetime
from pprint import pprint
import os

# Baseline
from metrics.snr import si_snr
from data.dataloader import DNSAudio

# Custom
from network import *
from helper import *
from fourier_transform import FourierTransform
from config import config


TRAINING = True
VALIDATION = False

OFFSET = 1.5 # used for scaling factors to avoid division by 0

print("Starting DNS Transformer")

# Parameters
parser = argparse.ArgumentParser(
                    prog='Transformer-based denoising',
                    description='This program can be used to explore different tranformer-based speech enhancement methods using parameters that are specified in the config file',
                    epilog='Refer to config.py to see examples, run the program as\npython main.py --config configuration_key')

parser.add_argument('-c', '--config', required=True, help="configuration - key in config dictionary in config.py")

args = parser.parse_args()
tag = args.config
print("Using Experiment with tag ", tag)
pprint(config[tag])

# Check config
assert config[tag]['device'] == "cuda" or config[tag]['device'] == "cpu", "Device must be 'cuda' or 'cpu'"
assert config[tag]['network'] == "generative" or config[tag]['network'] == "scaling", "Network must be 'generative' or 'scaling'"
assert config[tag]['epochs'] > 0, "Number of epochs must be greater than 0"
assert config[tag]['batch'] > 0, "Batch size must be greater than 0"
assert config[tag]['phase'] == True or config[tag]['phase'] == False, "Phase must be set too boolean value"
assert config[tag]['optimizer'] == "SGD" or config[tag]['optimizer'] == "Adam", "Optimizer must be 'SGD' or 'Adam'"
assert (config[tag]['lr'] > 0 or config[tag]['lr'] == "baseline"), "Learning rate must be a positive float or 'baseline'"
assert config[tag]['momentum'] >= 0, "Momentum must be a non-negative float"
assert config[tag]['sample_rate'] > 0, "Sample rate must be greater than 0"
assert config[tag]['n_fft'] > 0 and config[tag]['n_fft'] & (config[tag]['n_fft'] - 1) == 0, "n_fft must be a positive integer power of 2"
assert config[tag]['frame_s'] > 0, "Frame duration must be greater than 0"
assert config[tag]['stride_s'] > 0, "Stride duration must be greater than 0"
assert config[tag]['loss_mse']['mode'] == "scale" or config[tag]['loss_mse']['mode'] == "frequency", "MSE loss mode must be 'scale' or 'frequency'"
assert config[tag]['loss_mse']['weight'] >= 0, "MSE loss weight must be non-negative"
assert config[tag]['loss_snr'] >= 0, "SNR loss weight must be non-negative"

# Set parameters
device = config[tag]['device']
network = config[tag]['network']
epochs = config[tag]['epochs']
batch = config[tag]['batch']
phase = config[tag]['phase']
optimizer = config[tag]['optimizer']
lr = config[tag]['lr']
momentum = config[tag]['momentum']
sample_rate = config[tag]['sample_rate']
n_fft = config[tag]['n_fft']
frame = int(sample_rate * config[tag]['frame_s'])
stride = int(sample_rate * config[tag]['stride_s'])
mse_mode = config[tag]['loss_mse']['mode']
mse_weight = config[tag]['loss_mse']['weight']
snr_weight = config[tag]['loss_snr']
transformer = config[tag]['transformer']

# limit number of CPU cores
if device == "cpu":
  torch.set_num_threads(32)

dataset_dir = "../dataset/datasets_fullband/"

if 'gpu01.ids.rwth-aachen.de' == os.uname()[1]:
  trained_dir = "../trained_model_gpu01"
else:
  trained_dir = "../trained_model"
assert os.path.isdir(trained_dir), "Directory " + trained_dir + " does not exist"

### Dataset ###

# Loading data
train_set = DNSAudio(root=dataset_dir + 'training_set/')
validation_set = DNSAudio(root=dataset_dir + 'validation_set/')

# Example output
noisy_audio, clean_audio, noise_audio, metadata = train_set.__getitem__(0)
print(metadata)
saveAudio("example_noisy.wav", noisy_audio)
saveAudio("example_clean.wav", clean_audio)
saveAudio("example_noise.wav", noise_audio)

def collate_fn(batch):
  noisy, clean, noise = [], [], []

  for sample in batch:
    noisy += [torch.FloatTensor(sample[0])]
    clean += [torch.FloatTensor(sample[1])]
    noise += [torch.FloatTensor(sample[2])]

  return torch.stack(noisy), torch.stack(clean), torch.stack(noise)


############################################################
################## Complete DNSModelCoeff ##################
############################################################

net = DNSModel(sample_rate, n_fft, frame, stride, device, batch, phase, transformer)

params = sum(p.numel() for p in net.parameters() if p.requires_grad)
train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("No. params: ", params)
print("No. trainable params: ", train_params)

net.to(device)
train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
validation_loader = DataLoader(validation_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

if optimizer == "SGD":
  dns_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
else:
  dns_optimizer = torch.optim.Adam(net.parameters(), lr=lr)

my_fourier = FourierTransform(n_fft, stride, frame)

for epoch in range(epochs):
  print("EPOCH: ", epoch)
  #training
  if TRAINING:
    print("TRAINING - Coeffs")
    total_loss = 0
    t_st = datetime.now()
    net.train()
    for i, (noisy, clean, noise) in enumerate(train_loader):
      if i % 100 == 0:
        print("{}/{}".format(i, len(train_loader)))

      noisy = noisy.to(device)
      clean = clean.to(device)

      src = my_fourier.stft(noisy)
      tgt = my_fourier.stft(clean)


      src_phase = src[1]
      src_mag = src[0]
      src_mag = src_mag/torch.max(torch.abs(src_mag))

      tgt_phase = tgt[1]
      tgt_mag = tgt[0]
      tgt_mag = tgt_mag/torch.max(torch.abs(tgt_mag))
      
      # debugging only - train on noisy only (also at output)
      #tgt_phase = src[1]
      #tgt_mag = src[0]
      #tgt_mag = src_mag/torch.max(torch.abs(src_mag))

      if phase:
        net_src = torch.cat((src_mag, src_phase), dim=1)
        net_tgt = torch.cat((tgt_mag, tgt_phase), dim=1)
      else:
        net_src = src_mag
        net_tgt = tgt_mag

      if network == "scaling":
        scaling_coeffs_mag = tgt_mag/(src_mag + OFFSET)
        if phase:
          phase_shift = tgt_phase - src_phase
          scaling_coeffs = torch.cat((scaling_coeffs_mag, phase_shift), dim=1)
        net_out = net(net_src, scaling_coeffs)

        if phase:
          out_mag, out_phase = torch.split(net_out, split_size_or_sections=[int(n_fft/2) + 1, int(n_fft/2) + 1], dim=1)
          out_mag = (src_mag[:,:,1:]+OFFSET) * out_mag 
          out_phase = src_phase[:, :, 1:] + out_phase

          output_spec = torch.cat((out_mag, out_phase), dim=1)
          if mse_mode == "scale":
            mse = F.mse_loss(scaling_coeffs[:,:,1:], net_out)
          else:
            mse = F.mse_loss(net_tgt[:,:,1:], output_spec)
        else:
          output_spec = (net_src[:,:,1:]+OFFSET)*mag_factors_coeffs_out
          out_mag = output_spec
          out_phase = src_phase[:, :, 1:]
          if mse_mode == "scale":
            mse = F.mse_loss(mag_factors_coeffs[:,:,1:], mag_factors_coeffs_out)
          else:
            mse = F.mse_loss(tgt_mag[:,:,1:], output_spec)
        output = my_fourier.istft((out_mag, out_phase))
        snr = torch.mean(si_snr(my_fourier.istft((tgt_mag[:,:,1:], tgt_phase[:,:,1:])), output))
        loss = mse_weight * mse + snr_weight * (100 - snr)
      else: # generative
        output_spec = net(net_src, net_tgt)
        if phase:
          out_mag, out_phase = torch.split(output_spec, split_size_or_sections=[int(n_fft/2) + 1, int(n_fft/2) + 1], dim=1)
          mse = F.mse_loss(net_tgt[:,:,1:], output_spec)
        else:
          out_mag = output_spec
          out_phase = src_phase[:, :, 1:]
          mse = F.mse_loss(net_tgt[:,:,1:], out_mag)
        output = my_fourier.istft((out_mag, out_phase))
        snr = torch.mean(si_snr(my_fourier.istft((tgt_mag[:,:,1:], tgt_phase[:,:,1:])), output))
        loss = mse_weight * mse + snr_weight * (100 - snr)
      
      tgt = net_tgt[:,:,1:].contiguous()

      assert torch.isnan(loss) == False
      
      print("s {}".format(snr.item()))
      print("m {}".format(mse.item()))
      print("l {}".format(loss.item()), flush=True)

      dns_optimizer.zero_grad()
      loss.backward()
      dns_optimizer.step()

      total_loss += loss.detach().item()
      if i == 0:
        saveAudio("d2ebug_coeffs_train_0_noisy_{}.wav".format(epoch), noisy[0], toCPU=True)
        saveAudio("d2ebug_coeffs_train_0_clean_{}.wav".format(epoch), clean[0], toCPU=True)
        saveAudio("d2ebug_coeffs_train_0_output_{}.wav".format(epoch), output[0].detach(), toCPU=True)
      if i == 1000:
        saveAudio("d2ebug_coeffs_train_1_noisy_{}.wav".format(epoch), noisy[0], toCPU=True)
        saveAudio("d2ebug_coeffs_train_1_clean_{}.wav".format(epoch), clean[0], toCPU=True)
        saveAudio("d2ebug_coeffs_train_1_output_{}.wav".format(epoch), output[0].detach(), toCPU=True)
      if i == 10000:
        saveAudio("d2ebug_coeffs_train_2_noisy_{}.wav".format(epoch), noisy[0], toCPU=True)
        saveAudio("d2ebug_coeffs_train_2_clean_{}.wav".format(epoch), clean[0], toCPU=True)
        saveAudio("d2ebug_coeffs_train_2_output_{}.wav".format(epoch), output[0].detach(), toCPU=True)
    print("Total loss: ", total_loss/len(train_loader))
    print("Time elapsed - train: ", datetime.now() - t_st)
  
    saveAudio("d2ebug_coeffs_train_3_noisy_{}.wav".format(epoch), noisy[0], toCPU=True)
    saveAudio("d2ebug_coeffs_train_3_clean_{}.wav".format(epoch), clean[0], toCPU=True)
    saveAudio("d2ebug_coeffs_train_3_output_{}.wav".format(epoch), output[0].detach(), toCPU=True)

    torch.save(net.state_dict(), "{}/m2odel_epoch_{}.pt".format(trained_folder, epoch))
    torch.save(dns_optimizer.state_dict(), "{}/o2ptimizer_epoch_{}.pt".format(trained_folder, epoch))

  # Validation
  if VALIDATION:
    print("VALIDATION - Coeffs")
    total_loss = 0
    t_st = datetime.now()
    net.eval()
    for i, (noisy, clean, noise) in enumerate(validation_loader):
      print("validation_loader")
      print(len(validation_loader), flush=True)

      with torch.no_grad():
        if i % 100 == 0:
          print("{}/{}".format(i, len(train_loader)))

        noisy = noisy.to(device)
        clean = clean.to(device)

        src = my_fourier.stft(noisy)
        tgt = my_fourier.stft(clean)

        src_phase = None
        tgt_phase = None
        src_phase = src[1]
        src = src[0]
        tgt_phase = tgt[1]
        tgt = tgt[0]

        #dec_in = torch.zeros((batch, 257, 1))
        coeffs_out = torch.zeros((batch, 257, 1)) # TODO use first noisy frame as best approximation that we have
        coeffs_out = coeffs_out.to(device)
        for _ in range(3001):
          #print("test", flush=True)
          next_item = net(src, coeffs_out, TRAIN=False)
          #for k in range(next_item.shape[2]):
          #  print(next_item[0,0,k])
          #print("next_item ", next_item.shape)

          # Concatenate previous input with predicted best word
          coeffs_out = torch.cat((coeffs_out, next_item[:,:,-1:]), dim=2)
          #dec_in = next_item
          #print("next_item ", next_item.shape)

        coeffs_out = coeffs_out[:,:,1:-1]
        output = my_fourier.istft(((src[:,:,1:]+OFFSET)*coeffs_out, src_phase[:, :, 1:]))
        
        tgt = tgt[:,:,1:].contiguous()

        snr = torch.mean(si_snr(my_fourier.istft((tgt, tgt_phase[:,:,1:])), output))
        #loss = lam * F.mse_loss(coeffs_out, tgt) + (100 - torch.mean(score))
        mse = F.mse_loss(coeffs[:,:,1:], coeffs_out)

        loss = mse
        
        assert torch.isnan(loss) == False
        
        #print()
        print("snr: {}".format(snr.item()))
        #print(F.mse_loss(noisy, clean))
        print("mse: {}".format(mse.item()))
        print("loss: {}".format(loss.item()), flush=True)

        total_loss += loss.detach().item()
        if i == 0:
          saveAudio("debug_val_0_noisy.wav", noisy[0], toCPU=True)
          saveAudio("debug_val_0_clean.wav", clean[0], toCPU=True)
          saveAudio("debug_val_0_output.wav", output[0].detach(), toCPU=True)
        if i == 1000:
          saveAudio("debug_val_1_noisy.wav", noisy[0], toCPU=True)
          saveAudio("debug_val_1_clean.wav", clean[0], toCPU=True)
          saveAudio("debug_val_1_output.wav", output[0].detach(), toCPU=True)
        if i == 10000:
          saveAudio("debug_val_2_noisy.wav", noisy[0], toCPU=True)
          saveAudio("debug_val_2_clean.wav", clean[0], toCPU=True)
          saveAudio("debug_val_2_output.wav", output[0].detach(), toCPU=True)
    print("Total loss: ", total_loss/len(train_loader))
    print("Time elapsed - val: ", datetime.now() - t_st)

    saveAudio("debug_val_res_noisy.wav", noisy[0], toCPU=True)
    saveAudio("debug_val_res_clean.wav", clean[0], toCPU=True)
    saveAudio("debug_val_res_output.wav", output[0].detach(), toCPU=True)


  print("Done")