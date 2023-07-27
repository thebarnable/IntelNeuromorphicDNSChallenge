import re
import matplotlib.pyplot as plt

with open("../genetic/run_genetic.out", "r") as file:
  file_content = file.read()

# Use regex to find all avr_snr values
avr_snr_values = re.findall(r'avg_snr:\s+(-?\d+\.\d+)', file_content)

# Convert the extracted values to numerical data
avr_snr_values = [float(val) for val in avr_snr_values]

# Plot the avr_snr values using matplotlib
plt.plot(avr_snr_values, marker='o')
plt.xlabel('Generation')
plt.ylabel('avr_snr')
plt.title('avr_snr values over generations')
plt.grid(True)
plt.show()