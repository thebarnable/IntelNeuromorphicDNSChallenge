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
from main_network import *
from conformer_lstm import *
from helper import *
from fourier_transform import FourierTransform
from conf_lstm_config import config
from main_network import execute


##### Parameters #####
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
assert config[tag]['n_fft'] > 0, "n_fft must be a positive integer"
assert config[tag]['frame_s'] > 0, "Frame duration must be greater than 0"
assert config[tag]['stride_s'] > 0, "Stride duration must be greater than 0"
assert config[tag]['loss_mse']['mode'] == "scale" or config[tag]['loss_mse']['mode'] == "frequency", "MSE loss mode must be 'scale' or 'frequency'"
assert config[tag]['loss_mse']['weight'] >= 0, "MSE loss weight must be non-negative"
assert config[tag]['loss_snr'] >= 0, "SNR loss weight must be non-negative"

execute(config[tag], tag)


#######################################################