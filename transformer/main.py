import torch
from data.dataloader import DNSAudio
from reference.reference import *
from network.network import *
import numpy as np
from datetime import datetime
from scipy.io import wavfile
from torch.utils.data import DataLoader
from metrics.snr import si_snr
import torch.nn.functional as F


print("Starting DNS Transformer")

device = "cuda"
epochs = 10
batch = 1
lr = 0.01
lam = 0.001 # lagrangian factor
trained_folder = "trained_model"
dataset_dir = "dataset/datasets_fullband/"
TRAIN = True
REFERENCE = False

sample_rate = 16000
n_fft = 512
frame = int(sample_rate*0.025)
stride = int(sample_rate*0.01)

# Loading data
train_set = DNSAudio(root=dataset_dir + 'training_set/')
validation_set = DNSAudio(root=dataset_dir + 'validation_set/')

class Helper():
  def __init__(self, n_fft, stride, frame, polar=False):
    self.n_fft = n_fft
    self.stride = stride
    self.frame = frame
    self.polar = polar

  def stft(self, values):
    stft = torch.stft(values,
                      n_fft = self.n_fft,
                      hop_length=self.stride,
                      win_length=self.frame,
                      window=None,
                      center=True,
                      pad_mode='reflect',
                      normalized=True,
                      onesided=True,
                      return_complex=False
                    )
    if self.polar:
      real = stft[:, :, :, 0]
      imaginary = stft[:, :, :, 1]
      absolute = torch.sqrt(real ** 2 + imaginary ** 2)
      phase = torch.atan2(imaginary, real)
      return (absolute, phase)
    return stft

  def istft(self, values):
    if self.polar:
      magnitude = values[0]
      phase = values[1]
      real = magnitude * torch.cos(phase)
      imaginary = magnitude * torch.sin(phase)
      values = torch.stack((real, imaginary), dim=-1)
        
    istft = torch.istft(values,
                      n_fft = self.n_fft,
                      hop_length=self.stride,
                      win_length=self.frame,
                      window=None,
                      center=True,
                      normalized=True,
                      onesided=True,
                      return_complex=False
                    )
    return istft


def saveAudio(file_name, audio, toCPU=False, sample_rate=16000):
  if toCPU:
    my_audio = audio.cpu()
  else:
    my_audio = audio
  my_audio = np.array(my_audio.flatten(), dtype=np.float64)
  normalized_audio = my_audio / np.abs(my_audio).max()
  scaled_audio = np.int16((np.array(normalized_audio) * 32767))
  wavfile.write("audio/" + file_name, sample_rate, scaled_audio)


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

if REFERENCE:
  ############################################################
  ###################### STFT (Ref) ##########################
  ############################################################
  print("------- Using STFT -------")
  train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
  validation_loader = DataLoader(validation_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
  net = STFT()
  params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  print("No. params: ", params)
  print("No. trainable params: ", train_params)
  net.to(device)

  t_st = datetime.now()
  for i, (noisy, clean, noise) in enumerate(validation_loader):
    net.eval()

    with torch.no_grad():
      noisy = noisy.to(device)
      clean = clean.to(device)
      
      dns_sig = net(clean)

      score = si_snr(dns_sig, clean)
      loss = lam * F.mse_loss(dns_sig, clean) + (100 - torch.mean(score))

      processed = i * validation_loader.batch_size
      total = len(validation_loader.dataset)
      time_elapsed = (datetime.now() - t_st).total_seconds()
      samples_sec = time_elapsed / \
        (i + 1) / validation_loader.batch_size
      header_list = [f'Valid: [{processed}/{total} '
                f'({100.0 * processed / total:.0f}%)]']
      print("\nStatus: ")
      print("si_snr", si_snr(noisy, clean))
      print("F", F.mse_loss(noisy, clean))
      print("score", score)
      print("F", F.mse_loss(dns_sig, clean))
      print("loss", loss)
      print("i", i)
      print("samples_sec", samples_sec)
      print("header_list", header_list)
      
      saveAudio("stft_net_noisy.wav", noisy, toCPU=True)
      saveAudio("stft_net_clean.wav", clean, toCPU=True)
      saveAudio("stft_dns_sig_clean.wav", dns_sig, toCPU=True)
      break

  ############################################################
  ##################### NetWaveGlow (Ref) #################### # not working currently, see sampling rate
  ############################################################
  print("------- Using NetWaveGlow -------")
  train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
  validation_loader = DataLoader(validation_set, batch_size=batch, shuffle=True, 
                                  collate_fn=collate_fn, num_workers=4, pin_memory=True)
  net = NetWaveGlow(device)
  params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  print("No. params: ", params)
  print("No. trainable params: ", train_params)
  net.to(device)

  t_st = datetime.now()
  for i, (noisy, clean, noise) in enumerate(validation_loader):
    net.eval()

    with torch.no_grad():
      noisy = noisy.to(device)
      clean = clean.to(device)
      
      dns_sig = net(clean)

      #score = si_snr(dns_sig, clean)
      #loss = lam * F.mse_loss(dns_sig, clean) + (100 - torch.mean(score))

      processed = i * validation_loader.batch_size
      total = len(validation_loader.dataset)
      time_elapsed = (datetime.now() - t_st).total_seconds()
      samples_sec = time_elapsed / \
        (i + 1) / validation_loader.batch_size
      header_list = [f'Valid: [{processed}/{total} '
                f'({100.0 * processed / total:.0f}%)]']
      print("\nStatus: ")
      #print("si_snr", si_snr(noisy, clean))
      #print("F", F.mse_loss(noisy, clean))
      #print("score", score)
      #print("F", F.mse_loss(dns_sig, clean))
      #print("loss", loss)
      print("i", i)
      print("samples_sec", samples_sec)
      print("header_list", header_list)
      
      saveAudio("glow_net_noisy.wav", noisy, toCPU=True)
      saveAudio("glow_net_clean.wav", clean, toCPU=True)
      saveAudio("glow_dns_sig_clean.wav", dns_sig, toCPU=True)
      break


  ############################################################
  ################ GriffinLim (Ref) ####################
  ############################################################
  print("------- Using GriffinLim -------")
  train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
  validation_loader = DataLoader(validation_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
  net = GriffinLim()
  params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  print("No. params: ", params)
  print("No. trainable params: ", train_params)
  net.to(device)

  t_st = datetime.now()
  for i, (noisy, clean, noise) in enumerate(validation_loader):
    net.eval()

    noisy = noisy.to(device)
    clean = clean.to(device)
    
    dns_sig = net(clean)

    score = si_snr(dns_sig, clean)
    loss = lam * F.mse_loss(dns_sig, clean) + (100 - torch.mean(score))

    processed = i * validation_loader.batch_size
    total = len(validation_loader.dataset)
    time_elapsed = (datetime.now() - t_st).total_seconds()
    samples_sec = time_elapsed / \
      (i + 1) / validation_loader.batch_size
    header_list = [f'Valid: [{processed}/{total} '
              f'({100.0 * processed / total:.0f}%)]']
    print("\nStatus: ")
    print("si_snr", si_snr(noisy, clean))
    print("F", F.mse_loss(noisy, clean))
    print("score", score)
    print("F", F.mse_loss(dns_sig, clean))
    print("loss", loss)
    print("i", i)
    print("samples_sec", samples_sec)
    print("header_list", header_list)
    
    saveAudio("griffin_net_noisy.wav", noisy, toCPU=True)
    saveAudio("griffin_net_clean.wav", clean, toCPU=True)
    saveAudio("griffin_dns_sig_clean.wav", dns_sig, toCPU=True)
    break
### End of refernce networks ###



############################################################
################## Complete DNSModelCoeff ##################
############################################################

print("------- Complete DNSModelCoeff -------")
POLAR = True
net = DNSModelCoeff(sample_rate, n_fft, frame, stride, device, batch, POLAR)
params = sum(p.numel() for p in net.parameters() if p.requires_grad)
train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("No. params: ", params)
print("No. trainable params: ", train_params)
net.to(device)
train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
validation_loader = DataLoader(validation_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

dns_optimizer = None
if TRAIN:
  dns_optimizer = torch.optim.SGD(net.parameters(), lr=lr)
  dns_optimizer = torch.optim.Adam(net.parameters(), lr=lr)

my_helper = Helper(n_fft, stride, frame, polar=POLAR)
OFFSET = 1.5
VALIDATION = False

for epoch in range(epochs):
  print("EPOCH: ", epoch)
  #training
  if TRAIN:
    print("TRAINING - Coeffs")
    total_loss = 0
    t_st = datetime.now()
    net.train()
    for i, (noisy, clean, noise) in enumerate(train_loader):
      if i % 100 == 0:
        print("{}/{}".format(i, len(train_loader)))

      noisy = noisy.to(device)
      clean = clean.to(device)

      src = my_helper.stft(noisy)
      tgt = my_helper.stft(clean)

      src_phase = None
      tgt_phase = None
      if POLAR:
        src_phase = src[1]
        src = src[0]
        tgt_phase = tgt[1]
        tgt = tgt[0]

      
      coeffs = tgt/(src + OFFSET)
      
      coeffs_out = net(src, coeffs)

      if POLAR:
        output = my_helper.istft(((src[:,:,1:]+OFFSET)*coeffs_out, src_phase[:, :, 1:]))
      else:
        output = my_helper.istft((src[:,:,1:,:]+OFFSET)*coeffs_out)

      if POLAR:
        tgt = tgt[:,:,1:].contiguous()
      else:
        tgt = tgt[:,:,1:,:].contiguous()

      snr = None
      if POLAR:
        snr = torch.mean(si_snr(my_helper.istft((tgt, tgt_phase[:,:,1:])), output))
      else:
        snr = torch.mean(si_snr(my_helper.istft(tgt), output))
      #loss = lam * F.mse_loss(coeffs_out, tgt) + (100 - torch.mean(score))
      
      #real_part_coeffs_out = coeffs_out[:, :, :, 0]
      #imaginary_part_coeffs_out = coeffs_out[:, :, :, 1]
      #absolute_value_coeffs_out = torch.sqrt(real_part_coeffs_out ** 2 + imaginary_part_coeffs_out ** 2)
  
      #real_part_tgt = tgt[:, :, :, 0]
      #imaginary_part_tgt = tgt[:, :, :, 1]
      #absolute_value_tgt = torch.sqrt(real_part_tgt ** 2 + imaginary_part_tgt ** 2)
      #mse = F.mse_loss(absolute_value_coeffs_out, absolute_value_tgt)

      #output_verif = my_helper.istft(((src+10)*coeffs)[:,:,1:,:])
      #snr_verif = torch.mean(si_snr(my_helper.istft(tgt), output_verif))
      #print("snr_verif: {}".format(snr_verif.item()))
      #snr_verif = torch.mean(si_snr(clean, clean))
      #print("snr_verif2: {}".format(snr_verif.item()))
      #snr_verif = torch.mean(si_snr(clean, noisy))
      #print("snr_verif3: {}".format(snr_verif.item()))

      if POLAR:
        mse = F.mse_loss(coeffs[:,:,1:], coeffs_out)
      else:
        mse = F.mse_loss(coeffs[:,:,1:,:], coeffs_out)
      loss = mse
      
      assert torch.isnan(loss) == False
      
      #print()
      print("snr: {}".format(snr.item()))
      #print(F.mse_loss(noisy, clean))
      print("mse: {}".format(mse.item()))
      print("loss: {}".format(loss.item()), flush=True)

      dns_optimizer.zero_grad()
      loss.backward()
      dns_optimizer.step()

      total_loss += loss.detach().item()
      if i == 0:
        saveAudio("debug_coeffs_train_0_noisy_{}.wav".format(epoch), noisy[0], toCPU=True)
        saveAudio("debug_coeffs_train_0_clean_{}.wav".format(epoch), clean[0], toCPU=True)
        saveAudio("debug_coeffs_train_0_output_{}.wav".format(epoch), output[0].detach(), toCPU=True)
      if i == 1000:
        saveAudio("debug_coeffs_train_1_noisy_{}.wav".format(epoch), noisy[0], toCPU=True)
        saveAudio("debug_coeffs_train_1_clean_{}.wav".format(epoch), clean[0], toCPU=True)
        saveAudio("debug_coeffs_train_1_output_{}.wav".format(epoch), output[0].detach(), toCPU=True)
      if i == 10000:
        saveAudio("debug_coeffs_train_2_noisy_{}.wav".format(epoch), noisy[0], toCPU=True)
        saveAudio("debug_coeffs_train_2_clean_{}.wav".format(epoch), clean[0], toCPU=True)
        saveAudio("debug_coeffs_train_2_output_{}.wav".format(epoch), output[0].detach(), toCPU=True)
    print("Total loss: ", total_loss/len(train_loader))
    print("Time elapsed - train: ", datetime.now() - t_st)
  
    saveAudio("debug_coeffs_train_3_noisy_{}.wav".format(epoch), noisy[0], toCPU=True)
    saveAudio("debug_coeffs_train_3_clean_{}.wav".format(epoch), clean[0], toCPU=True)
    saveAudio("debug_coeffs_train_3_output_{}.wav".format(epoch), output[0].detach(), toCPU=True)

    torch.save(net.state_dict(), "{}/model_epoch_{}.pt".format(trained_folder, epoch))
    torch.save(dns_optimizer.state_dict(), "{}/optimizer_epoch_{}.pt".format(trained_folder, epoch))

  # Validation
  # TODO non-polar check
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

        src = my_helper.stft(noisy)
        tgt = my_helper.stft(clean)

        src_phase = None
        tgt_phase = None
        if POLAR:
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

        if POLAR:
          coeffs_out = coeffs_out[:,:,1:-1]
          output = my_helper.istft(((src[:,:,1:]+OFFSET)*coeffs_out, src_phase[:, :, 1:]))
        else:
          output = my_helper.istft((src[:,:,1:,:]+OFFSET)*coeffs_out)

        if POLAR:
          tgt = tgt[:,:,1:].contiguous()
        else:
          tgt = tgt[:,:,1:,:].contiguous()

        snr = None
        if POLAR:
          snr = torch.mean(si_snr(my_helper.istft((tgt, tgt_phase[:,:,1:])), output))
        else:
          snr = torch.mean(si_snr(my_helper.istft(tgt), output))
        #loss = lam * F.mse_loss(coeffs_out, tgt) + (100 - torch.mean(score))
        
        #real_part_coeffs_out = coeffs_out[:, :, :, 0]
        #imaginary_part_coeffs_out = coeffs_out[:, :, :, 1]
        #absolute_value_coeffs_out = torch.sqrt(real_part_coeffs_out ** 2 + imaginary_part_coeffs_out ** 2)
    
        #real_part_tgt = tgt[:, :, :, 0]
        #imaginary_part_tgt = tgt[:, :, :, 1]
        #absolute_value_tgt = torch.sqrt(real_part_tgt ** 2 + imaginary_part_tgt ** 2)
        #mse = F.mse_loss(absolute_value_coeffs_out, absolute_value_tgt)

        #output_verif = my_helper.istft(((src+10)*coeffs)[:,:,1:,:])
        #snr_verif = torch.mean(si_snr(my_helper.istft(tgt), output_verif))
        #print("snr_verif: {}".format(snr_verif.item()))
        #snr_verif = torch.mean(si_snr(clean, clean))
        #print("snr_verif2: {}".format(snr_verif.item()))
        #snr_verif = torch.mean(si_snr(clean, noisy))
        #print("snr_verif3: {}".format(snr_verif.item()))

        if POLAR:
          mse = F.mse_loss(coeffs[:,:,1:], coeffs_out)
        else:
          mse = F.mse_loss(coeffs[:,:,1:,:], coeffs_out)
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



############################################################
################## Complete DNSModelBasic ##################
############################################################

if False:
  print("------- Complete DNSModelBasic -------")
  net = DNSModelBasic(sample_rate, n_fft, frame, stride, device, batch)
  params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
  print("No. params: ", params)
  print("No. trainable params: ", train_params)
  net.to(device)
  train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
  validation_loader = DataLoader(validation_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

  dns_optimizer = None
  if TRAIN:
    dns_optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    #dns_optimizer = torch.optim.Adam(net.parameters(), lr=lr)

  my_helper = Helper(n_fft, stride, frame)

  for epoch in range(epochs):

    #training
    if TRAIN:
      print("TRAINING")
      total_loss = 0
      t_st = datetime.now()
      net.train()
      for i, (noisy, clean, noise) in enumerate(train_loader):
        if i % 100 == 0:
          print("{}/{}".format(i, len(train_loader)))

        noisy = noisy.to(device)
        clean = clean.to(device)

        src = my_helper.stft(noisy)
        tgt = my_helper.stft(clean)
        spec_out = net(src, tgt)

        # Add first frame unchanged at beginning
        #spec_out = [src[:,:,:1,:], spec_out]
        #spec_out = torch.cat(spec_out, dim=2)

        output = my_helper.istft(spec_out)
        tgt = tgt[:,:,1:,:].contiguous()

        score = torch.mean(si_snr(my_helper.istft(tgt), output))
        #loss = lam * F.mse_loss(spec_out, tgt) + (100 - torch.mean(score))
        
        real_part_spec_out = spec_out[:, :, :, 0]
        imaginary_part_spec_out = spec_out[:, :, :, 1]
        absolute_value_spec_out = torch.sqrt(real_part_spec_out ** 2 + imaginary_part_spec_out ** 2)
    
        real_part_tgt = tgt[:, :, :, 0]
        imaginary_part_tgt = tgt[:, :, :, 1]
        absolute_value_tgt = torch.sqrt(real_part_tgt ** 2 + imaginary_part_tgt ** 2)

        mse = F.mse_loss(absolute_value_spec_out, absolute_value_tgt)

        loss = 0.001 * mse + (100 - score)
        
        assert torch.isnan(loss) == False

        dns_optimizer.zero_grad()
        loss.backward()
        dns_optimizer.step()

        
        #print()
        print("snr: {}".format(score.item()))
        #print(F.mse_loss(noisy, clean))
        print("mse: {}".format(mse.item()))
        print("loss: {}".format(loss.item()), flush=True)
        total_loss += loss.detach().item()
        if i == 0:
          saveAudio("debug_train_0_noisy.wav", noisy[0], toCPU=True)
          saveAudio("debug_train_0_clean.wav", clean[0], toCPU=True)
          saveAudio("debug_train_0_output.wav", output[0].detach(), toCPU=True)
        if i == 1000:
          saveAudio("debug_train_1_noisy.wav", noisy[0], toCPU=True)
          saveAudio("debug_train_1_clean.wav", clean[0], toCPU=True)
          saveAudio("debug_train_1_output.wav", output[0].detach(), toCPU=True)
        if i == 10000:
          saveAudio("debug_train_2_noisy.wav", noisy[0], toCPU=True)
          saveAudio("debug_train_2_clean.wav", clean[0], toCPU=True)
          saveAudio("debug_train_2_output.wav", output[0].detach(), toCPU=True)
      print("Total loss: ", total_loss/len(train_loader))
      print("Time elapsed - train: ", datetime.now() - t_st)
    
      saveAudio("debug_train_res_noisy.wav", noisy[0], toCPU=True)
      saveAudio("debug_train_res_clean.wav", clean[0], toCPU=True)
      saveAudio("debug_train_res_output.wav", output[0].detach(), toCPU=True)

    # Validation
    print("VALIDATION")
    total_loss = 0
    t_st = datetime.now()
    net.eval()
    for i, (noisy, clean, noise) in enumerate(validation_loader):

      with torch.no_grad():
        if i % 100 == 0:
          print("{}/{}".format(i, len(train_loader)))

        noisy = noisy.to(device)
        clean = clean.to(device)

        src = my_helper.stft(noisy)
        tgt = my_helper.stft(clean)

        dec_in = torch.zeros((batch, 1, 512))
        dec_in = dec_in.to(device)
        for _ in range(3001):
          print("test", flush=True)
          next_item = net(src, dec_in, TRAIN=False)
          print("next_item ", next_item.shape)

          # Concatenate previous input with predicted best word
          dec_in = torch.cat((dec_in, next_item), dim=1)
        
        # Add first frame unchanged at beginning
        spec_out = [src[:,:,:1,:], dec_in]
        spec_out = torch.cat(spec_out, dim=2)

        output = my_helper.istft(spec_out)

        score = torch.mean(si_snr(clean, output))
        #loss = lam * F.mse_loss(spec_out, tgt) + (100 - torch.mean(score))
        
        real_part_spec_out = spec_out[:, :, :, 0]
        imaginary_part_spec_out = spec_out[:, :, :, 1]
        absolute_value_spec_out = torch.sqrt(real_part_spec_out ** 2 + imaginary_part_spec_out ** 2)
    
        real_part_tgt = tgt[:, :, :, 0]
        imaginary_part_tgt = tgt[:, :, :, 1]
        absolute_value_tgt = torch.sqrt(real_part_tgt ** 2 + imaginary_part_tgt ** 2)

        mse = F.mse_loss(absolute_value_spec_out, absolute_value_tgt)

        loss = 0.001 * mse + (100 - score)
        
        #print()
        print("snr: {}".format(score.item()))
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