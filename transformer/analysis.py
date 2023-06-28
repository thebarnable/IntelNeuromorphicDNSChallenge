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

import matplotlib.pyplot as plt


print("Starting Analysis")

device = "cpu"
epochs = 1 #10
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


def collate_fn(batch):
  noisy, clean, noise = [], [], []

  for sample in batch:
    noisy += [torch.FloatTensor(sample[0])]
    clean += [torch.FloatTensor(sample[1])]
    noise += [torch.FloatTensor(sample[2])]

  return torch.stack(noisy), torch.stack(clean), torch.stack(noise)

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
saveAudio("analysis_noisy.wav", noisy_audio)
saveAudio("analysis_clean.wav", clean_audio)
saveAudio("analysis_noise.wav", noise_audio)


############################################################
######################## anaylsis ##########################
############################################################


train_loader = DataLoader(train_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
validation_loader = DataLoader(validation_set, batch_size=batch, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

POLAR = True
my_helper = Helper(n_fft, stride, frame, polar=POLAR)

train_min = 0
train_max = 0

for i, (noisy, clean, noise) in enumerate(train_loader):
  if i % 100 == 0:
    print("{}/{}".format(i, len(train_loader)))


  noise = my_helper.stft(noise)
  src = my_helper.stft(noisy)
  tgt = my_helper.stft(clean)

  noise_phase = None
  src_phase = None
  tgt_phase = None
  if POLAR:
    src_phase = src[1]
    src = src[0]
    tgt_phase = tgt[1]
    tgt = tgt[0]
    noise_phase = noise[1]
    noise = noise[0]

  src_scaled = src/torch.max(torch.abs(src))
  tgt_scaled = tgt/torch.max(torch.abs(tgt))

  ### Heatmap of SNRs ###
  if False:
    if i == 0 or i == 10:
      specs = [noise, src, tgt]
      snr_combinations = []
      for a in specs:
        row = []
        for b in specs:
          row.append(si_snr(my_helper.istft(a), my_helper.istft(b)))
        snr_combinations.append(row)
      print(snr_combinations, flush=True)

      snr_combinations_array = np.array([[tensor.item() for tensor in row] for row in snr_combinations])
      fig, ax = plt.subplots()
      heatmap = ax.imshow(snr_combinations_array, cmap='viridis')
      for i in range(3):
        for j in range(3):
          ax.annotate(str(snr_combinations_array[i, j]), xy=(j, i), ha='center', va='center', color='white')
      cbar = plt.colorbar(heatmap)
      ax.set_xticks(np.arange(3))
      ax.set_yticks(np.arange(3))
      ax.set_xticklabels(['noise', 'src', 'noisy'])
      ax.set_yticklabels(['noise', 'src', 'noisy'])
      ax.set_title('SNRs')
      plt.show()

  ### Distributions per tensor ###
  if False:
    if i == 0:

      coeffs_0 = (tgt/10)/(src/10+1)
      coeffs_1 = tgt/(src+1)
      coeffs_2 = tgt/(src+2)
      coeffs_3 = tgt/(src+3)
      coeffs_4 = tgt/(src+4)
      min_noise = torch.min(noise)
      min_src = torch.min(src)
      min_tgt = torch.min(tgt)
      max_noise = torch.max(noise)
      max_src = torch.max(src)
      max_tgt = torch.max(tgt)
      min_coeffs_0 = torch.min(coeffs_0)
      max_coeffs_0 = torch.max(coeffs_0)
      min_coeffs_1 = torch.min(coeffs_1)
      max_coeffs_1 = torch.max(coeffs_1)
      min_coeffs_2 = torch.min(coeffs_2)
      max_coeffs_2 = torch.max(coeffs_2)
      min_coeffs_3 = torch.min(coeffs_3)
      max_coeffs_3 = torch.max(coeffs_3)
      min_coeffs_4 = torch.min(coeffs_4)
      max_coeffs_4 = torch.max(coeffs_4)

    
      mins = [min_noise.item(), min_src.item(), min_tgt.item(), min_coeffs_0.item(), min_coeffs_1.item(), min_coeffs_2.item(), min_coeffs_3.item(), min_coeffs_4.item()]
      maxs = [max_noise.item(), max_src.item(), max_tgt.item(), max_coeffs_0.item(), max_coeffs_1.item(), max_coeffs_2.item(), max_coeffs_3.item(), max_coeffs_4.item()]

      print("min noise: ", min_noise)
      print("min src: ", min_src)
      print("min tgt: ", min_tgt)
      print("max noise: ", max_noise)
      print("max src: ", max_src)
      print("max tgt: ", max_tgt)
      print("min coeffs_0: ", min_coeffs_0)
      print("max coeffs_0: ", max_coeffs_0)
      print("min coeffs_1: ", min_coeffs_1)
      print("max coeffs_1: ", max_coeffs_1)
      print("min coeffs_2: ", min_coeffs_2)
      print("max coeffs_2: ", max_coeffs_2)
      print("min coeffs_3: ", min_coeffs_3)
      print("max coeffs_3: ", max_coeffs_3)
      print("min coeffs_4: ", min_coeffs_4)
      print("max coeffs_4: ", max_coeffs_4)
      
      src_array = src.flatten().numpy()
      tgt_array = tgt.flatten().numpy()
      noise_array = noise.flatten().numpy()
      coeffs_0_array = coeffs_0.flatten().numpy()
      coeffs_1_array = coeffs_1.flatten().numpy()
      coeffs_2_array = coeffs_2.flatten().numpy()
      coeffs_3_array = coeffs_3.flatten().numpy()
      coeffs_4_array = coeffs_4.flatten().numpy()

      fig, ax = plt.subplots()

      tot_min = min(mins)
      tot_max = max(maxs)
      # Plot histograms
      #ax.hist(src_array, range=(tot_min, tot_max), bins=50, label='src', color='red', alpha=0.5)
      #ax.hist(tgt_array, range=(tot_min, tot_max), bins=50, label='tgt', color='blue', alpha=0.5)
      #ax.hist(noise_array, range=(tot_min, tot_max), bins=50, label='noise', color='green', alpha=0.5)
      #ax.hist(coeffs_0_array, range=(tot_min, tot_max), bins=50, label='coeffs_0', color='orange', alpha=0.5)
      #ax.hist(coeffs_1_array, range=(tot_min, tot_max), bins=50, label='coeffs_1', color='cyan', alpha=0.5)
      #ax.hist(coeffs_2_array, range=(tot_min, tot_max), bins=50, label='coeffs_2', color='yellow', alpha=0.5)
      #ax.hist(coeffs_3_array, range=(tot_min, tot_max), bins=50, label='coeffs_3', color='purple', alpha=0.5)
      #ax.hist(coeffs_4_array, range=(tot_min, tot_max), bins=50, label='coeffs_4', color='brown', alpha=0.5)

      ax.hist(coeffs_0_array, range=(tot_min, tot_max), bins=1000, label='coeffs_0', color='orange', alpha=0.5)

      # Set labels and title
      ax.set_xlabel('Value')
      ax.set_ylabel('Frequency')
      ax.set_title('Distribution of Tensors')

      # Add legend
      ax.legend()

      # Display the plot
      plt.show()

  ### Total max/min ###
  if False:
    min_src = torch.min(src)
    min_tgt = torch.min(tgt)
    max_src = torch.max(src)
    max_tgt = torch.max(tgt)
    train_min = min([min_src.item(), min_tgt.item(), train_min])
    train_max = max([max_src.item(), max_tgt.item(), train_max])
    # total_max  13.71932601928711 -> trainset
    # total_min  -12.511946678161621 -> trainset

  ### STFT ###
  if False:
    if i < 10:
      i_tgt = my_helper.istft(tgt)
      i_tgt_10 = my_helper.istft(tgt/10)
      print("snr_clean_clean", si_snr(clean, clean))
      print("snr_i_tgt_clean", si_snr(i_tgt, clean))
      print("snr_i_tgt_clean", si_snr(i_tgt_10, clean))
      saveAudio("anaylsis_snr_clean.wav", clean[0], toCPU=True)
      saveAudio("anaylsis_snr_i_tgt.wav", i_tgt[0], toCPU=True)
      saveAudio("anaylsis_snr_i_tgt_10.wav", i_tgt_10[0], toCPU=True)

  ### Coeffs ###
  if True:
    if i < 1:
      offset = 1.5
      coeffs_scaled = tgt_scaled/(src_scaled + offset)

      #print(tgt_scaled[0, 0, 0, 0])
      #print(src_scaled[0, 0, 0, 0])
      #print(coeffs_scaled[0, 0, 0, 0])
      #print(tgt_scaled[0, 2, 3, 1])
      #print(src_scaled[0, 2, 3, 1])
      #print(coeffs_scaled[0, 2, 3, 1])

      #print("src_max: ", torch.max(torch.abs(src)))
      #print("scaled_src_max: ", torch.max(torch.abs(src_scaled)))
      #print("tgt_max: ", torch.max(torch.abs(tgt)))
      #print("scaled_tgt_max: ", torch.max(torch.abs(tgt_scaled)))
      #print("scaled_coeffs_max: ", torch.max(torch.abs(coeffs_scaled)))
      
      print("Boundaries")
      if POLAR:
        i_tgt = my_helper.istft((tgt, tgt_phase))
        i_tgt_scaled = my_helper.istft((tgt_scaled, tgt_phase))
        back_trans = coeffs_scaled * (src_scaled + offset)
        i_coeffs = my_helper.istft((back_trans, src_phase))
        print("snr_max_by_abs", si_snr(my_helper.istft((tgt, src_phase)), clean).item())
        print("snr_verify", si_snr(my_helper.istft((tgt, tgt_phase)), clean).item())
      else:
        i_tgt = my_helper.istft(tgt)
        i_tgt_scaled = my_helper.istft(tgt_scaled)
        back_trans = coeffs_scaled * (src_scaled + offset)
        i_coeffs = my_helper.istft(back_trans)

      print("snr_clean_clean", si_snr(clean, clean).item())
      print("snr_i_tgt_clean", si_snr(i_tgt, clean).item())
      print("snr_i_coeffs_clean", si_snr(i_coeffs, clean).item())
      random_coeff = torch.tensor(np.random.normal(0, 0.01, coeffs_scaled.size()))
      ran_coeffs_back = random_coeff * (src_scaled + offset)
      if POLAR:
        i_random_coeff = my_helper.istft((ran_coeffs_back, src_phase))
      else:
        i_random_coeff = my_helper.istft(ran_coeffs_back)
      random_snr = si_snr(i_random_coeff, clean)
      loss = 0.001 * F.mse_loss(ran_coeffs_back, tgt_scaled) + (100 - torch.mean(random_snr))
      print("snr_random", random_snr.item())
      print("mse_random", F.mse_loss(random_coeff, coeffs_scaled).item())
      print("loss_random", loss.item())
      print("def_loss_random", F.mse_loss(random_coeff, coeffs_scaled).item())
      saveAudio("anaylsis_snr_clean.wav", clean[0], toCPU=True)
      saveAudio("anaylsis_snr_i_tgt.wav", i_tgt_scaled[0], toCPU=True)
      saveAudio("anaylsis_snr_i_coeffs_clean.wav", i_coeffs[0], toCPU=True)

      exit(1)
      # Plot histogram
      if False and i == 0:
        flattened = coeffs_scaled.flatten().numpy()
        fig, ax = plt.subplots()
        ax.hist(flattened, bins=150, label='coeffs_scaled', color='orange', alpha=1)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of coeffs_scaled')
        ax.legend()

        plt.figure(2)
        flattened = src_scaled.flatten().numpy()
        fig, ax = plt.subplots()
        ax.hist(flattened, bins=150, label='src_scaled', color='orange', alpha=1)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of src_scaled')
        ax.legend()

        plt.figure(3)
        flattened = tgt_scaled.flatten().numpy()
        fig, ax = plt.subplots()
        ax.hist(flattened, bins=150, label='tgt_scaled', color='orange', alpha=1)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of tgt_scaled')
        ax.legend()
        plt.show()

      num_iterations = 150

      print("------------ Std sweep ------------")
      std_sweep_mse = []
      std_sweep_snr = []
      stds = []
      for i in range(num_iterations):
        mean = 0
        std = 0.01 * i
        stds.append(std)
        coeffs_noise = coeffs_scaled + torch.tensor(np.random.normal(mean, std, coeffs_scaled.size()), dtype=torch.float)
        mse = F.mse_loss(coeffs_noise, coeffs_scaled)
        if POLAR:
          i_coeffs = my_helper.istft((coeffs_noise * (src_scaled + offset), src_phase))
        else:
          i_coeffs = my_helper.istft(coeffs_noise * (src_scaled + offset))
        snr = si_snr(i_coeffs, clean).item()
        print("mean: {} \t std: {}\t mse: {}\t snr: {}".format(mean, std, mse, snr))
        std_sweep_mse.append(mse)
        std_sweep_snr.append(snr)
      fig, ax1 = plt.subplots()
      ax1.plot(stds, std_sweep_mse, color='blue')
      ax1.set_xlabel('Std')
      ax1.set_ylabel('MSE', color='blue')
      ax1.tick_params('y', colors='blue')
      ax2 = ax1.twinx()
      ax2.plot(stds, std_sweep_snr, color='red')
      ax2.set_ylabel('SNR', color='red')
      ax2.tick_params('y', colors='red')
      plt.title('Std Sweep Analysis')

      # TODO check why  
      print("------------ Mean sweep ------------")
      plt.figure(1)
      mean_sweep_mse = []
      mean_sweep_snr = []
      means = []
      for i in range(num_iterations):
        mean = 0.01 * i
        means.append(mean)
        std = 0.02
        coeffs_noise = coeffs_scaled + torch.tensor(np.random.normal(mean, std, coeffs_scaled.size()), dtype=torch.float)
        mse = F.mse_loss(coeffs_noise, coeffs_scaled)
        if POLAR:
          i_coeffs = my_helper.istft((coeffs_noise * (src_scaled + offset), src_phase))
        else:
          i_coeffs = my_helper.istft(coeffs_noise * (src_scaled + offset))
        snr = si_snr(i_coeffs, clean).item()
        print("mean: {} \t std: {}\t mse: {}\t snr: {}".format(mean, std, mse, snr))
        mean_sweep_mse.append(mse)
        mean_sweep_snr.append(snr)
      fig, ax1 = plt.subplots()
      ax1.plot(means, mean_sweep_mse, color='blue')
      ax1.set_xlabel('Mean')
      ax1.set_ylabel('MSE', color='blue')
      ax1.tick_params('y', colors='blue')
      ax2 = ax1.twinx()
      ax2.plot(means, mean_sweep_snr, color='red')
      ax2.set_ylabel('SNR', color='red')
      ax2.tick_params('y', colors='red')
      plt.title('Mean Sweep Analysis')

      print("------------ Shift sweep ------------")
      plt.figure(2)
      shift_sweep_mse = []
      shift_sweep_snr = []    
      shifts = []
      for i in range(num_iterations):
        shift = 0.01 * i
        shifts.append(shift)
        coeffs_noise = coeffs_scaled + shift
        mse = F.mse_loss(coeffs_noise, coeffs_scaled)
        if POLAR:
          i_coeffs = my_helper.istft((coeffs_noise * (src_scaled + offset), src_phase))
        else:
          i_coeffs = my_helper.istft(coeffs_noise * (src_scaled + offset))
        snr = si_snr(i_coeffs, clean).item()
        print("shift: {}\t mse: {}\t snr: {}".format(shift, mse, snr))
        shift_sweep_mse.append(mse)
        shift_sweep_snr.append(snr)
      fig, ax1 = plt.subplots()
      ax1.plot(shifts, shift_sweep_mse, color='blue')
      ax1.set_xlabel('Shift')
      ax1.set_ylabel('MSE', color='blue')
      ax1.tick_params('y', colors='blue')
      ax2 = ax1.twinx()
      ax2.plot(shifts, shift_sweep_snr, color='red')
      ax2.set_ylabel('SNR', color='red')
      ax2.tick_params('y', colors='red')
      plt.title('Shift Sweep Analysis')
      plt.show()

      #print("------------ Mult sweep ------------")
      #mult_sweep_mse = []
      #mult_sweep_snr = []    
      #for i in range(num_iterations):
      #  factor = 0.01 * i
      #  coeffs_noise = coeffs_scaled * factor
      #  mse = F.mse_loss(coeffs_noise, coeffs_scaled)
      #  i_coeffs = my_helper.istft(coeffs_noise * (src_scaled + offset))
      #  snr = si_snr(i_coeffs, clean).item()
      #  print("factor: {}\t mse: {}\t snr: {}".format(factor, mse, snr))
      #  mult_sweep_mse.append(mse)
      #  mult_sweep_snr.append(snr)
      saveAudio("anaylsis_snr_i_coeffs_noise.wav", i_coeffs[0], toCPU=True)


  if True and i == 11:
    break

print("total_max ", train_max)
print("total_min ", train_min)


#print("snr: {}".format(snr.item()))
#print("mse: {}".format(mse.item()))
#print("loss: {}".format(loss.item()), flush=True)



#saveAudio("anaylsis_noisy.wav", noisy[0], toCPU=True)
#saveAudio("anaylsis_clean.wav", clean[0], toCPU=True)
#saveAudio("anaylsis_output.wav", output[0].detach(), toCPU=True)

print("Done")

