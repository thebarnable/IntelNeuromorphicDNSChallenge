import re
import matplotlib.pyplot as plt

# Read the file
with open("testing_dns_transformer.out", "r") as file:
#with open("dns_transformer.out", "r") as file:
    content = file.read()

# Extract the values from the last tensor in each entry
loss_pattern = r"loss: ([\d.e-]+)"
mse_pattern = r"mse: ([\d.e-]+)"
snr_pattern = r"snr: ([\d.e-]+)"
loss_values = re.findall(loss_pattern, content)
mse_values = re.findall(mse_pattern, content)
snr_values = re.findall(snr_pattern, content)

# Convert the values to float
loss_values = [float(value) for value in loss_values if float(value) < 30] # TODO, random high number in between
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
ax1.set_xlabel('Entry')
ax1.legend(loc='upper left')
ax2.legend(loc='upper center')
ax3.legend(loc='upper right')

# Display the plot
plt.show()
