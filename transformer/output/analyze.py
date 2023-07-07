import re
import matplotlib.pyplot as plt
import os

MOVING_AVERAGE = True
WINDOW_SIZE = 400


#def mov_avg(my_list):
#  output = []
#  length = len(my_list)
#  for i in range(length):
#    if i + WINDOW_SIZE > length:
#      break
#    values = [float(value) for value in my_list[i:i + WINDOW_SIZE]]  # Convert to float
#    output.append(sum(values) / WINDOW_SIZE)
#  return output

import numpy as np

def mov_avg(my_list):
  my_array = np.array(my_list, dtype=float)
  cum_sum = np.cumsum(my_array)
  cum_sum[WINDOW_SIZE:] = cum_sum[WINDOW_SIZE:] - cum_sum[:-WINDOW_SIZE]
  return cum_sum[WINDOW_SIZE - 1:] / WINDOW_SIZE


def show_figure(file_name):
  # Read the file
  with open(file_name, "r") as file:
    content = file.read()

  # Extract the values from the last tensor in each entry
  loss_pattern = r"l ([\d.e+-]+)"
  mse_pattern = r"m ([\d.e+-]+)"
  snr_pattern = r"s ([\d.e+-]+)"

  loss_values_found = re.findall(loss_pattern, content)
  mse_values_found = re.findall(mse_pattern, content)
  snr_values_found = re.findall(snr_pattern, content)
  if MOVING_AVERAGE:
    loss_values = mov_avg(loss_values_found)
    mse_values = mov_avg(mse_values_found)
    snr_values = mov_avg(snr_values_found)
  else:
    loss_values = [float(value) for value in loss_values_found]
    mse_values = [float(value) for value in mse_values_found]
    snr_values = [float(value) for value in snr_values_found]

  # Create figure and axes
  fig, ax1 = plt.subplots()

  # Plot loss values
  ax1.plot(range(len(loss_values)), loss_values, color='tab:red', label='Loss')
  ax1.set_xlabel('Batch')
  ax1.set_ylabel('Loss', color='tab:red')

  # Create a twin y-axis for MSE
  ax2 = ax1.twinx()
  ax2.plot(range(len(mse_values)), mse_values, color='tab:blue', label='MSE')
  ax2.set_ylabel('MSE', color='tab:blue')

  # Create a twin y-axis for SNR
  ax3 = ax1.twinx()
  ax3.spines['right'].set_position(('outward', 60))  # Adjust the position of the SNR axis
  ax3.plot(range(len(snr_values)), snr_values, color='tab:green', label='SNR')
  ax3.set_ylabel('SNR', color='tab:green')

  fig.suptitle(file_name)

  # Set the labels and legend
  ax1.legend(loc='upper left')
  ax2.legend(loc='upper center')
  ax3.legend(loc='upper right')


all_files_baseline = [
  "prev/scaling_0.out",
  "prev/scaling_1_frequency_loss.out",
  "prev/scaling_2_SGD.out",
  "prev/scaling_3_no_phase.out",
  "prev/generative_0.out",
  "prev/generative_2_SGD.out",
  "prev/generative_3_no_phase.out",

  "prev/scaling_mse_0.out",
  "prev/scaling_mse_1_frequency_loss.out",
  "prev/scaling_mse_2_SGD.out",
  "prev/scaling_mse_3_no_phase.out",
  "prev/generative_mse_0.out",
  "prev/generative_mse_2_SGD.out",
  "prev/generative_mse_3_no_phase.out",

  "prev/scaling_snr_0.out",
  "prev/scaling_snr_1_frequency_loss.out",
  "prev/scaling_snr_2_SGD.out",
  "prev/scaling_snr_3_no_phase.out",
  "prev/generative_snr_0.out",
  "prev/generative_snr_2_SGD.out",
  "prev/generative_snr_3_no_phase.out"
]

all_files_scaling_2 = [
  "scaling_2_SGD_large.out",
  "scaling_2_SGD_snr0_2.out",
  "scaling_2_SGD_mse0_5.out",
  "scaling_2_SGD_e3.out",
  "scaling_2_SGD_large_e3.out"
]

all_files_fft = [
  "scaling_2_SGD_fft_1024.out",
  "scaling_2_SGD_fft_128.out",
  "scaling_2_SGD_05_02.out"
]

local_explore = [
  "scaling_2_SGD_fft_256.out",
  "scaling_2_SGD_fft_1024_05_02.out",
  "scaling_2_SGD_e10.out",
  "scaling_2_e10.out",
  "scaling_2_SGD_mse0_5_e10.out" 
]

# TODO fix NameError: name 'scaling_coeffs' is not defined
tmp = [
  "scaling_snr_3_no_phase.out",
  "scaling_3_no_phase.out",
  "scaling_2_SGD_e10_e30.out" 
]

first = True
for i, item in enumerate(tmp): # TODO choose list
  print("Analyzing ", item)
  if os.path.exists(item):
    if first:
      first = False
    else:
      plt.figure(i)
    show_figure(item)
  else:
    print("File {} doesn't exist".format(item))
plt.show()
