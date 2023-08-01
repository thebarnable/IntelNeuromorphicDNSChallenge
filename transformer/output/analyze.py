import re
import matplotlib.pyplot as plt
import os
from pprint import pprint

MOVING_AVERAGE = True
WINDOW_SIZE = 200
MERGE = True


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
  return list(cum_sum[WINDOW_SIZE - 1:] / WINDOW_SIZE)
  #test = cum_sum[WINDOW_SIZE - 1:] / WINDOW_SIZE
  #ret_val = [test[item] for item in range(len(test)) if item % 30000 == 0]
  #ret_val = test[::WINDOW_SIZE]
  #return test


def show_figure(file_name):

  if not type(file_name) == list: # case MERGE
    file_name = [file_name]

  loss_values = []
  mse_values = []
  snr_values = []
  for item in file_name:
    # Read the file
    with open(item, "r") as file:
      content = file.read()

    # Extract the values from the last tensor in each entry
    loss_pattern = r"l ([\d.e+-]+)"
    mse_pattern = r"m ([\d.e+-]+)"
    snr_pattern = r"s ([\d.e+-]+)"

    loss_values_found = re.findall(loss_pattern, content)
    mse_values_found = re.findall(mse_pattern, content)
    snr_values_found = re.findall(snr_pattern, content)
    if len(loss_values_found) == 0: # case of validation:
      # Extract the values from the last tensor in each entry
      loss_pattern = r"l_v ([\d.e+-]+)"
      mse_pattern = r"m_v ([\d.e+-]+)"
      snr_pattern = r"s_v ([\d.e+-]+)"

      loss_values_found = re.findall(loss_pattern, content)
      mse_values_found = re.findall(mse_pattern, content)
      snr_values_found = re.findall(snr_pattern, content)
    if MOVING_AVERAGE:
      loss_values += mov_avg(loss_values_found)
      mse_values += mov_avg(mse_values_found)
      snr_values += mov_avg(snr_values_found)
    else:
      loss_values += [float(value) for value in loss_values_found]
      mse_values += [float(value) for value in mse_values_found]
      snr_values += [float(value) for value in snr_values_found]

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

  ret = -9999
  if len(snr_values) > 0:
    ret = snr_values[int(len(snr_values)/4)]#-1]
  return ret


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
  "prev/scaling_2_SGD_fft_256.out",
  "prev/scaling_2_SGD_fft_1024_05_02.out",
  "prev/scaling_2_SGD_e10.out",
  "prev/scaling_2_e10.out",
  "prev/scaling_2_SGD_mse0_5_e10.out" 
]

tmp = [
  "prev/scaling_2_SGD_e10_e30.out" ,
  "prev/scaling_2_SGD_no_phase.out",
  "prev/scaling_2_SGD_lr0_1.out"
]

scaling_2_SGD = [
  "prev/scaling_2_SGD_e10.out",
  "prev/scaling_2_SGD_e10_e30.out" ,
  "scaling_2_SGD_e30_e50.out",
]

conformer = [
  "conformer_1_01.out",
  "conformer_lr05.out",
  "conformer_05_01.out",
  "conformer_1_02.out",
  "conformer_1_03.out",
  "conformer_1_1.out",
]

conf_layer = [
  "conformer_l4.out",
  "conformer_l6.out",
  "conformer_l8.out",
  "conformer_l10.out"
]

conf_l6 = [
  "conformer_l6_no_phase.out",
  "conformer_l6_Adam.out",
  "conformer_l6_mse_freq.out",
  "conformer_l6_gen.out",
  "conformer_l6_gen_Adam.out",
  "conformer_l6_gen_no_phase.out",
  "conformer_l6_gen_lr02.out",
  "conformer_l6_gen_e10.out"
]


conf_size = [
  "conformer_l6_h4.out",
  "conformer_l6_kernel_15.out",
  "conformer_l4_h4.out",
  "conformer_l4_kernel_15.out",
  "conformer_l4_h4_kernel_15.out",
  "conformer_l4_h4_kernel_15_dff128.out",
  "conformer_l4_h2_kernel_15_dff128.out",
  "conformer_l4_h4_kernel_7_dff128.out",
  "conformer_l4_h4_kernel_7_dff64.out"
]

conf_explore = [
  "conformer_l6_no_phase.out",
  "conformer_l6_Adam.out",
  "conformer_l6_mse_freq.out",
  "conformer_l6_gen.out",
  "conformer_l6_gen_Adam.out",
  "conformer_l6_gen_no_phase.out",
  "conformer_l6_gen_lr02.out",
  "conformer_l6_gen_e10.out",
  "conformer_l4.out",
  "conformer_l6.out",
  "conformer_l8.out",
  "conformer_l10.out",
  "conformer_1_01.out",
  "conformer_05_01.out",
  "conformer_1_02.out",
  "conformer_1_03.out",
  "conformer_1_1.out"
]

conformer_lstm = [
  "conformer_lstm.out",
  "conformer_lstm_l4.out",
  "conformer_lstm_l6.out",
  "conformer_lstm_l8.out",
  "conformer_lstm_l10.out",
  "conformer_lstm_1_01.out",
  "prev/conformer_lstm_lr05.out",
  "conformer_lstm_05_01.out",
  "conformer_lstm_1_02.out",
  "conformer_lstm_1_03.out",
  "conformer_lstm_1_1.out",
  "conformer_lstm_l6_no_phase.out",
  "conformer_lstm_l6_Adam.out",
  "conformer_lstm_l6_mse_freq.out",
  "conformer_lstm_l6_kernel_15.out",
  "conformer_lstm_l6_h4.out",
  "conformer_lstm_l6_gen.out",
  "conformer_lstm_l6_gen_Adam.out",
  "conformer_lstm_l6_gen_no_phase.out",
  "conformer_lstm_l6_gen_lr02.out",
  "conformer_lstm_l4_h4.out",
  "conformer_lstm_l4_kernel_15.out",
  "conformer_lstm_l4_h4_kernel_15.out",
  "conformer_lstm_l4_h4_kernel_15_dff128.out",
  "conformer_lstm_l4_h2_kernel_15_dff128.out",
  "conformer_lstm_l4_h4_kernel_7_dff128.out",
  "conformer_lstm_l4_h4_kernel_7_dff64.out"
]

new_conf_lstm_promising = [
  "conformer_lstm_lr05_e2.out",
  "conformer_lstm_lr1.out",
  "prev/conformer_lstm_lr05_snr02.out",
  "conformer_lstm_lr_05_mse05.out",
  "conformer_lstm_l6_mse_freq.out",
  "conformer_lstm_l6_mse_freq_lr05.out",
  "conformer_lstm_l6_mse_freq_snr02.out",
  "conformer_lstm_l6_mse_freq_mse05.out"
]

new_conf_lstm_explore = [
  "conformer_lstm.out",
  "conformer_lstm_l4_h4.out",
  "conformer_lstm_l4_kernel_15.out",
  "conformer_lstm_l4_h4_kernel_15.out",
  "conformer_lstm_l4_h4_kernel_15_dff128.out",
  "conformer_lstm_l4_h2_kernel_15_dff128.out",
  "conformer_lstm_l4_h4_kernel_7_dff128.out",
  "conformer_lstm_l4_h4_kernel_7_dff64.out",
  "conformer_lstm_l6_no_phase.out",
  "conformer_lstm_l6_Adam.out",
  "conformer_lstm_l6_mse_freq.out",
  "conformer_lstm_l6_kernel_15.out",
  "conformer_lstm_l6_h4.out"
]

new_conf_lstm_explore_2 = [
  "conformer_lstm_l6_gen.out",
  "conformer_lstm_l6_gen_Adam.out",
  "conformer_lstm_l6_gen_no_phase.out",
  "conformer_lstm_l6_gen_lr02.out",
  "conformer_lstm_l4.out",
  "conformer_lstm_l6.out",
  "conformer_lstm_l8.out",
  "conformer_lstm_l10.out",
  "conformer_lstm_1_01.out",
  "prev/conformer_lstm_lr05.out",
  "conformer_lstm_05_01.out",
  "conformer_lstm_1_02.out",
  "conformer_lstm_1_03.out",
  "conformer_lstm_1_1.out"
]

new_conf_lstm_explore_local = [
  "conformer_lstm_lr05_snr02_lr01.out",
  "conformer_lstm_lr05_snr02_contd_e4.out",
  "conformer_lstm_lr05_0_03.out"
]

conformer_lstm_lr05_snr02_lr01 = [
  "prev/conformer_lstm_lr05_snr02.out",
  "conformer_lstm_lr05_snr02_lr01.out"
]

conformer_lstm_lr05_snr02_contd_e4 = [
  "prev/conformer_lstm_lr05_snr02.out",
  "conformer_lstm_lr05_snr02_contd_e4.out"
]

my_list = new_conf_lstm_promising + new_conf_lstm_explore + new_conf_lstm_explore_2  # TODO choose list
my_list = new_conf_lstm_explore_local
#my_list = ["conformer_lstm_lr05_snr02.out", "conformer_lstm_lr05.out", "conformer_lstm_lr_05_mse05.out"]
my_list = ["genetic_init_1000.out", "genetic_init_1000_e2.out", "genetic_init_1000_lr02_e2_5.out"]


last_values = []
if MERGE:
  show_figure(my_list)
else:
  first = True
  for i, item in enumerate(my_list):
    print("Analyzing ", item)
    if os.path.exists(item):
      if first:
        first = False
      else:
        plt.figure(i)
      last_snr = show_figure(item)
      last_values.append((last_snr, item))
    else:
      print("File {} doesn't exist".format(item))

print("\nSorted by last SNR")
pprint(sorted(last_values))

plt.show()