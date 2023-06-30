import re
import matplotlib.pyplot as plt
import os

def show_figure(file_name):
  # Read the file
  with open(file_name, "r") as file:
    content = file.read()

  # Extract the values from the last tensor in each entry
  loss_pattern = r"l ([\d.e-]+)"
  mse_pattern = r"m ([\d.e-]+)"
  snr_pattern = r"s ([\d.e-]+)"
  loss_values = re.findall(loss_pattern, content)
  mse_values = re.findall(mse_pattern, content)
  snr_values = re.findall(snr_pattern, content)

  # Convert the values to float
  loss_values = [float(value) for value in loss_values]
  mse_values = [float(value) for value in mse_values]
  snr_values = [float(value) for value in snr_values]

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

  # Set the labels and legend
  ax1.legend(loc='upper left')
  ax2.legend(loc='upper center')
  ax3.legend(loc='upper right')


all_files = [
  "scaling_0.out",
  "scaling_1_frequency_loss.out",
  "scaling_2_SGD.out",
  "scaling_3_no_phase.out",
  "generative_0.out",
  "generative_2_SGD.out",
  "generative_3_no_phase.out"

  "scaling_mse_0.out",
  "scaling_mse_1_frequency_loss.out",
  "scaling_mse_2_SGD.out",
  "scaling_mse_3_no_phase.out",
  "generative_mse_0.out",
  "generative_mse_2_SGD.out",
  "generative_mse_3_no_phase.out"

  "scaling_snr_0.out",
  "scaling_snr_1_frequency_loss.out",
  "scaling_snr_2_SGD.out",
  "scaling_snr_3_no_phase.out",
  "generative_snr_0.out",
  "generative_snr_2_SGD.out",
  "generative_snr_3_no_phase.out"
]

first = True
for item in all_files:
  if os.path.exists(item):
    if first:
      first = False
    else:
      plt.figure()
    show_figure(item)
  else:
    print("File {} doesn't exist".format(item))
  plt.show()
