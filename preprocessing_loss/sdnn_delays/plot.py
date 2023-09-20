import matplotlib.pyplot as plt
import numpy as np
from math import log2

def plot1():
    # Define the data
    data = [
        (2, 32, 8.44), (3, 32, 8.45), (4, 32, 8.64), (5, 32, 8.71),
        (6, 32, 8.02), (7, 32, 7.73), (8, 32, 8.26), (9, 32, 8.01),
        (2, 64, 9.49), (3, 64, 9.45), (4, 64, 9.53), (5, 64, 9.65),
        (6, 64, 8.8),  (7, 64, 8.08), (8, 64, 8.91), (9, 64, 8.51),
        (2, 128, 9.83), (3, 128, 10.11), (4, 128, 10.08), (5, 128, 10.17),
        (6, 128, 9.25), (7, 128, 9.12), (8, 128, 8.96), (9, 128, 9.13)
    ]

    # Separate data into x, y, and color values
    x = [item[0] for item in data]
    y = [item[1] for item in data]
    colors = [item[2] for item in data]

    # Create a grid of 3 rows and 9 columns
    grid = np.zeros((3, 8))

    # Fill the grid with color values based on x and y coordinates
    for i in range(len(data)):
        x_val, y_val, color_val = data[i]
        row = int(log2(y_val)-5)  # Determine the row based on y-coordinate
        col = x_val - 2    # Determine the column based on x-coordinate
        grid[row, col] = color_val

    # Create a discrete heatmap
    plt.figure(figsize=(5,5))
    cax = plt.matshow(grid, cmap='Blues', fignum=1) # vmin= 7.63, vmax=12.71,
    plt.colorbar(cax, label='SI-SNR')
    plt.xticks(range(8), range(2, 10))
    plt.yticks(range(3), [32, 64, 128])
    plt.xlabel('Number of Layers')
    plt.ylabel('Channel Width')
    # plt.title('Discrete Heatmap')

    # Show the plot
    #plt.show()
    plt.savefig('plot1.svg', transparent=True)

def plot2():
    # Define the data
    data = [
        (0.1, "Adadelta", 11.82),
        (0.1, "Adagrad", 11.92),
        (0.1, "Adam", 9.9),
        (0.1, "AdamW", 7.63),
        (0.01, "Adadelta", 10.53),
        (0.01, "Adagrad", 11.53),
        (0.01, "Adam", 11.72),
        (0.01, "AdamW", 11.18),
        (1e-3, "Adadelta", 7.99),
        (1e-3, "Adagrad", 10.73),
        (1e-3, "Adam", 12.57),
        (1e-3, "AdamW", 12.71),
        (1e-4, "Adadelta", 3.97),
        (1e-4, "Adagrad", 7.73),
        (1e-4, "Adam", 12.18),
        (1e-4, "AdamW", 12.15),
        (1e-5, "Adadelta", -4.53),
        (1e-5, "Adagrad", 0.98),
        (1e-5, "Adam", 10.02),
        (1e-5, "AdamW", 10.1),
    ]

    # Extract unique learning rates and optimizers
    learning_rates = sorted(list(set(item[0] for item in data)))
    optimizers = sorted(list(set(item[1] for item in data)))

    # Create a grid to store color values
    grid = np.zeros((len(learning_rates), len(optimizers)))

    # Fill the grid with color values based on the data
    for i in range(len(data)):
        lr_index = learning_rates.index(data[i][0])
        opt_index = optimizers.index(data[i][1])
        grid[lr_index, opt_index] = float(data[i][2])

    # Create a discrete heatmap
    plt.figure(figsize=(5,5))
    cax = plt.matshow(grid, cmap='Blues', fignum=2) #vmin= 7.63, vmax=12.71,
    plt.colorbar(cax, label='SI-SNR')
    plt.xticks(np.arange(len(optimizers)), optimizers, rotation=45)
    plt.yticks(np.arange(len(learning_rates)), learning_rates)
    #plt.xlabel('Optimizer')
    plt.ylabel('Learning Rate')
    #plt.title('Discrete Heatmap')
    plt.gca().xaxis.tick_top()  # Move x-axis ticks to the top for readability

    # Show the plot
    #plt.show()
    plt.savefig('plot2.svg', transparent=True)

if __name__ == '__main__':
    plot1()
    plot2()