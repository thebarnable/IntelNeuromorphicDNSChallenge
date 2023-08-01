import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.io import wavfile
import numpy as np
from datetime import datetime
from pprint import pprint
import os
import traceback

# Baseline
from metrics.snr import si_snr
from data.dataloader import DNSAudio

# Custom
from network import *
from conformer_lstm import *
from helper import *
from fourier_transform import FourierTransform
from conf_lstm_config import config




def execute(input_dict, tag, genetic=False, limit=1000):

  print("Starting DNS Transformer")

  OFFSET = 1 # used for scaling factors to avoid division by 0
  no_gpus = torch.cuda.device_count()
  print("CUDA devices: ", no_gpus)

  VALIDATION = False
  if ("validation" in input_dict):
    if input_dict["validation"]:
      print("VALIDATION")
      VALIDATION = True
  TRAINING = (not VALIDATION)
  if TRAINING:
    print("TRAINING")

  VALIDATION = (not TRAINING)

  if 'gpu01.ids.rwth-aachen.de' == os.uname()[1]:
    trained_dir = "../trained_model_gpu01/"
  else:
    trained_dir = "../trained_model/"
    if genetic:
      trained_dir = "../genetic/"
  assert os.path.isdir(trained_dir), "Directory " + trained_dir + " does not exist"


  load_pretrained = ("pretrained" in input_dict)
  if load_pretrained:
    pretrained_dir = trained_dir + "prev/" + input_dict["pretrained"] + "/"
    assert os.path.isdir(pretrained_dir), pretrained_dir + " does not exist"
    assert "pretrained_epoch" in input_dict, "pretrained_epoch is not in the configuration"
    assert input_dict['pretrained_epoch'] > 0, "Number of epochs must be greater than 0"
    pretrained_epoch = input_dict['pretrained_epoch']
  else:
    pretrained_epoch = 0

  # Set parameters
  device = input_dict['device']
  network = input_dict['network']
  epochs = input_dict['epochs']
  batch = input_dict['batch']
  phase = input_dict['phase']
  optimizer = input_dict['optimizer']
  lr = input_dict['lr']
  momentum = input_dict['momentum']
  sample_rate = input_dict['sample_rate']
  n_fft = input_dict['n_fft']
  frame = int(sample_rate * input_dict['frame_s'])
  stride = int(sample_rate * input_dict['stride_s'])
  mse_mode = input_dict['loss_mse']['mode']
  mse_weight = input_dict['loss_mse']['weight']
  snr_weight = input_dict['loss_snr']
  transformer = input_dict['transformer']
  if "conformer" in transformer:
    use_conformer = transformer['conformer']
    print("Using CONFORMER")
  else:
    use_conformer = False

  # limit number of CPU cores
  if device == "cpu":
    torch.set_num_threads(32)

  dataset_dir = "../dataset/datasets_fullband/"

  # Check that output directories for this experiment do not yet exist and create them
  # audio
  #assert os.path.isdir("../audio/" + tag) == False, "../audio/" + tag + " already exists"
  if os.path.isdir("../audio/" + tag) == False:
    os.makedirs("../audio/" + tag)
  # trained models
  my_trained_dir = trained_dir + tag
  assert os.path.isdir(my_trained_dir) == False, my_trained_dir + " already exists"

  ### Dataset ###

  # Loading data
  train_set = DNSAudio(root=dataset_dir + 'training_set/')
  validation_set = DNSAudio(root=dataset_dir + 'validation_set/')

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
  my_fourier = FourierTransform(n_fft, stride, frame)


  train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)
  validation_loader = DataLoader(validation_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=8, pin_memory=True)

  val_loader_iter = iter(validation_loader)
  example_data, example_target, example_noise = next(val_loader_iter)
  example = my_fourier.stft(example_data)

  if use_conformer:
    net = DNSModelConformer(sample_rate, n_fft, frame, stride, device, batch, phase, transformer, length=example[0].shape[2], no_gpus=no_gpus)
  else:
    net = DNSModel(sample_rate, n_fft, frame, stride, device, batch, phase, transformer, no_gpus=no_gpus)
  net = nn.DataParallel(net.to(device))

  if load_pretrained:
    net.load_state_dict(torch.load("{}/model_epoch_{}.pt".format(pretrained_dir, pretrained_epoch-1)))

  params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  print("No. params: ", params)
  print("No. trainable params: ", train_params, flush=True)


  if optimizer == "SGD":
    dns_optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  else:
    dns_optimizer = torch.optim.Adam(net.parameters(), lr=lr)
  if load_pretrained:
    dns_optimizer.load_state_dict(torch.load("{}/optimizer_epoch_{}.pt".format(pretrained_dir, pretrained_epoch-1)))


  
  avg_window_size = 100
  last_snrs = np.full(avg_window_size, -999, dtype=float)
  avg_index = 0
  avg_snr = None

  lengths = torch.Tensor([example[0].shape[2]] * batch)

  if use_conformer:
    shift = 0
  else:
    shift = 1

  try:
    for this_epoch in range(epochs):
      epoch = this_epoch + pretrained_epoch
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
            else:
              scaling_coeffs = scaling_coeffs_mag
            if use_conformer:
              net_out = net(net_src, lengths)
            else:
              net_out = net(net_src, scaling_coeffs)

            if phase:
              out_mag, out_phase = torch.split(net_out, split_size_or_sections=[int(n_fft/2) + 1, int(n_fft/2) + 1], dim=1)
              out_mag = (src_mag[:,:,shift:]+OFFSET) * out_mag 
              out_phase = src_phase[:, :, shift:] + out_phase

              output_spec = torch.cat((out_mag, out_phase), dim=1)
              if mse_mode == "scale":
                mse = F.mse_loss(scaling_coeffs[:,:,shift:], net_out)
              else:
                mse = F.mse_loss(net_tgt[:,:,shift:], output_spec)
            else:
              output_spec = (net_src[:,:,shift:]+OFFSET)*net_out
              out_mag = output_spec
              out_phase = src_phase[:, :, shift:]
              if mse_mode == "scale":
                mse = F.mse_loss(scaling_coeffs[:,:,shift:], net_out)
              else:
                mse = F.mse_loss(tgt_mag[:,:,shift:], output_spec)
            output = my_fourier.istft((out_mag, out_phase))
            snr = torch.mean(si_snr(my_fourier.istft((tgt_mag[:,:,shift:], tgt_phase[:,:,shift:])), output))
            loss = mse_weight * mse + snr_weight * (100 - snr)
          else: # generative
            if use_conformer:
              output_spec = net(net_src, lengths)
            else:
              output_spec = net(net_src, net_tgt)
            if phase:
              out_mag, out_phase = torch.split(output_spec, split_size_or_sections=[int(n_fft/2) + 1, int(n_fft/2) + 1], dim=1)
              mse = F.mse_loss(net_tgt[:,:,shift:], output_spec)
            else:
              out_mag = output_spec
              out_phase = src_phase[:, :, shift:]
              mse = F.mse_loss(net_tgt[:,:,shift:], out_mag)
            output = my_fourier.istft((out_mag, out_phase))
            snr = torch.mean(si_snr(my_fourier.istft((tgt_mag[:,:,shift:], tgt_phase[:,:,shift:])), output))
            loss = mse_weight * mse + snr_weight * (100 - snr)
          
          tgt = net_tgt[:,:,shift:].contiguous()

          assert torch.isnan(loss) == False
          
          print("s {}".format(snr.item()), flush=True)
          if not genetic:
            print("m {}".format(mse.item()))
            print("l {}".format(loss.item()), flush=True)

          # avg snr for genetic fitness
          last_snrs[avg_index] = snr
          avg_index = (avg_index + 1) % avg_window_size
          avg_snr = np.mean(last_snrs)

          dns_optimizer.zero_grad()
          loss.backward()
          dns_optimizer.step()

          total_loss += loss.detach().item()
          if not genetic:
            if i == 0:
              saveAudio("/{}/e{}_i{}_train_noisy.wav".format(tag, epoch, i), noisy[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_train_clean.wav".format(tag, epoch, i), clean[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_train_output.wav".format(tag, epoch, i), output[0].detach(), toCPU=True)
            if i == 1000:
              saveAudio("/{}/e{}_i{}_train_noisy.wav".format(tag, epoch, i), noisy[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_train_clean.wav".format(tag, epoch, i), clean[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_train_output.wav".format(tag, epoch, i), output[0].detach(), toCPU=True)
            if i == 10000:
              saveAudio("/{}/e{}_i{}_train_noisy.wav".format(tag, epoch, i), noisy[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_train_clean.wav".format(tag, epoch, i), clean[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_train_output.wav".format(tag, epoch, i), output[0].detach(), toCPU=True)
          
          if genetic:
            if i == limit:
              break
        print("Total loss: ", total_loss/len(train_loader))
        print("Time elapsed - train: ", datetime.now() - t_st)
      
      
        if not genetic:
          saveAudio("/{}/e{}_i{}e_train_noisy.wav".format(tag, epoch, i), noisy[0], toCPU=True)
          saveAudio("/{}/e{}_i{}e_train_clean.wav".format(tag, epoch, i), clean[0], toCPU=True)
          saveAudio("/{}/e{}_i{}e_train_output.wav".format(tag, epoch, i), output[0].detach(), toCPU=True)

        if os.path.isdir(my_trained_dir) == False:
          os.makedirs(my_trained_dir)
        
        if genetic:
          pass
          #torch.save(net.state_dict(), "{}/model_epoch.pt".format(my_trained_dir))
          #torch.save(dns_optimizer.state_dict(), "{}/optimizer_epoch.pt".format(my_trained_dir))
        else:
          torch.save(net.state_dict(), "{}/model_epoch_{}.pt".format(my_trained_dir, epoch))
          torch.save(dns_optimizer.state_dict(), "{}/optimizer_epoch_{}.pt".format(my_trained_dir, epoch))

      # Validation
      if VALIDATION:
        print("VALIDATION - Coeffs")
        total_loss = 0
        t_st = datetime.now()
        net.eval()
        for i, (noisy, clean, noise) in enumerate(validation_loader):

          with torch.no_grad():
            if i % 100 == 0:
              print("{}/{}".format(i, len(validation_loader)))

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
              else:
                scaling_coeffs = scaling_coeffs_mag
              if use_conformer:
                net_out = net(net_src, lengths)
              else:
                net_out = net(net_src, scaling_coeffs)
                #print(phase_shift, flush=True)
                #net_out = net(net_src, scaling_coeffs)
                #coeffs_out = scaling_coeffs[:,:,:1]
                for i in range(3000):
                  next_item = net(net_src, coeffs_out, TRAIN=False)
                  coeffs_out = torch.cat((coeffs_out, next_item[:,:,-shift:]), dim=2)
                  #print(coeffs_out, flush=True)
                  #coeffs_out = torch.cat((coeffs_out, scaling_coeffs[:,:,i+shift:i+2]), dim=2)
                net_out = coeffs_out[:,:,shift:]
                torch.set_printoptions(threshold=1000)
                np.set_printoptions(threshold=1000)
                #print(net_out)

              if phase:
                out_mag, out_phase = torch.split(net_out, split_size_or_sections=[int(n_fft/2) + 1, int(n_fft/2) + 1], dim=1)
                out_mag = (src_mag[:,:,shift:]+OFFSET) * out_mag 
                out_phase = src_phase[:, :, shift:] + out_phase

                output_spec = torch.cat((out_mag, out_phase), dim=1)
                if mse_mode == "scale":
                  mse = F.mse_loss(scaling_coeffs[:,:,shift:], net_out)
                else:
                  mse = F.mse_loss(net_tgt[:,:,shift:], output_spec)
              else:
                output_spec = (net_src[:,:,shift:]+OFFSET)*net_out
                out_mag = output_spec
                out_phase = src_phase[:, :, shift:]
                if mse_mode == "scale":
                  mse = F.mse_loss(scaling_coeffs[:,:,shift:], net_out)
                else:
                  mse = F.mse_loss(tgt_mag[:,:,shift:], output_spec)
              output = my_fourier.istft((out_mag, out_phase))
              snr = torch.mean(si_snr(my_fourier.istft((tgt_mag[:,:,shift:], tgt_phase[:,:,shift:])), output))
              loss = mse_weight * mse + snr_weight * (100 - snr)
            else: # generative
              if use_conformer:
                output_spec = net(net_src, lengths)
              else:
                #output_spec = net(net_src, net_tgt)
                for _ in range(3000):
                  next_item = net(net_src, coeffs_out, TRAIN=False)
                  coeffs_out = torch.cat((coeffs_out, next_item[:,:,-shift:]), dim=2)
                output_spec = coeffs_out[:,:,shift:]

              if phase:
                out_mag, out_phase = torch.split(output_spec, split_size_or_sections=[int(n_fft/2) + 1, int(n_fft/2) + 1], dim=1)
                mse = F.mse_loss(net_tgt[:,:,shift:], output_spec)
              else:
                out_mag = output_spec
                out_phase = src_phase[:, :, shift:]
                mse = F.mse_loss(net_tgt[:,:,shift:], out_mag)
              output = my_fourier.istft((out_mag, out_phase))
              snr = torch.mean(si_snr(my_fourier.istft((tgt_mag[:,:,shift:], tgt_phase[:,:,shift:])), output))
              loss = mse_weight * mse + snr_weight * (100 - snr)
            
            tgt = net_tgt[:,:,shift:].contiguous()

            assert torch.isnan(loss) == False
            
            print("s_v {}".format(snr.item()))
            print("m_v {}".format(mse.item()))
            print("l_v {}".format(loss.item()), flush=True)

            total_loss += loss.detach().item()
            if i == 0:
              saveAudio("/{}/e{}_i{}_val_noisy.wav".format(tag, epoch, i), noisy[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_val_clean.wav".format(tag, epoch, i), clean[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_val_output.wav".format(tag, epoch, i), output[0].detach(), toCPU=True)
            if i == 1000:
              saveAudio("/{}/e{}_i{}_val_noisy.wav".format(tag, epoch, i), noisy[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_val_clean.wav".format(tag, epoch, i), clean[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_val_output.wav".format(tag, epoch, i), output[0].detach(), toCPU=True)
            if i == 10000:
              saveAudio("/{}/e{}_i{}_val_noisy.wav".format(tag, epoch, i), noisy[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_val_clean.wav".format(tag, epoch, i), clean[0], toCPU=True)
              saveAudio("/{}/e{}_i{}_val_output.wav".format(tag, epoch, i), output[0].detach(), toCPU=True)
        print("Total loss: ", total_loss/len(validation_loader))
        print("Time elapsed - val: ", datetime.now() - t_st)
      
        saveAudio("/{}/e{}_i{}e_val_noisy.wav".format(tag, epoch, i), noisy[0], toCPU=True)
        saveAudio("/{}/e{}_i{}e_val_clean.wav".format(tag, epoch, i), clean[0], toCPU=True)
        saveAudio("/{}/e{}_i{}e_val_output.wav".format(tag, epoch, i), output[0].detach(), toCPU=True)
      
      print("avg_snr: ", avg_snr)
      if genetic:
        print("Done")
        return avg_snr
  except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        # Handle the CUDA out of memory error here
        print("CUDA out of memory error caught.")
        return -9999
    else:
        # Handle other types of runtime errors
        print("An unexpected runtime error occurred:", e)
        traceback.print_exc()
        return None
  return None
