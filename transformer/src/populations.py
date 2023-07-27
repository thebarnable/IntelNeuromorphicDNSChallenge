import numpy as np
from pprint import pprint

# using this baselinee as a kick_off point
#  'conformer_lstm_1_1' : {
#    "device" : "cuda",
#    "network" : "scaling",
#    "epochs" : 2,
#    "batch" : 2,
#    "phase" : True,
#    "optimizer" : "SGD",
#    "lr" : 0.01,
#    "momentum" : 0,
#    "sample_rate" : 16000,
#    "n_fft" : 512,
#    "frame_s" : 0.025,
#    "stride_s" : 0.01,
#    "loss_mse" :  {
#                    "mode" : "scale",
#                    "weight" : 1
#                  },
#    "loss_snr" : 1,
#    "transformer" : {
#                      "conformer" : True,
#                      "d_model" : 256,
#                      "heads" : 8,
#                      "d_ff" : 256,
#                      "dropout" : 0.1,
#                      "num_layers" : 12,
#                      "depthwise_conv_kernel_size" : 31
#                    }
#  }

### input vector ### only lower limits are hard
# lr           = 0.01    [0.001, 0.1] 
# n_fft        = 512     [32, 1024] 
# frame_s      = 0.025   [0.02, 0.032] 
# stride_s     = 0.01    [0.01, 0.032] 
# mse_weight   = 1       [0, 10]
# snr_weight   = 1       [0, 10]
# d_model      = 256     [32, 512] 
# heads        = 8       [2, 128]
# d_ff         = 256     [32, 512] 
# num_layers   = 12      [1, 16]
# kernel_size  = 31      [3, 63]
# lstm_layers  = 1       [1, 5]

def get_ranges():
  return np.array([
            [0.001, 0.1],
            [32, 1024],
            [0.02, 0.032],
            [0.01, 0.032],
            [0, 10],
            [0, 10],
            [32, 512],
            [2, 128],
            [32, 512],
            [1, 16],
            [3, 63],
            [1, 5]
          ])
      
def get_init_abs():
  return np.array([
            0.01,
            512,
            0.025,
            0.01,
            1,
            1,
            256,
            8,
            256,
            12,
            31,
            1
          ])


def abs_to_pop(abs):
  ranges = get_ranges()
  return (abs - ranges[:,0]) / (ranges[:,1] - ranges[:,0])

def fix_pop(population):
  sample_rate = 16000

  ranges = get_ranges()

  fixed = []
  for sample in population:
    lr          = sample[0]  * (ranges[0][1]  - ranges[0][0]) + ranges[0][0]
    n_fft       = int(round(sample[1]  * (ranges[1][1]  - ranges[1][0]) + ranges[1][0]))
    frame_s     = sample[2]  * (ranges[2][1]  - ranges[2][0]) + ranges[2][0]
    stride_s    = sample[3]  * (ranges[3][1]  - ranges[3][0]) + ranges[3][0]
    mse_weight  = sample[4]  * (ranges[4][1]  - ranges[4][0]) + ranges[4][0]
    snr_weight  = sample[5]  * (ranges[5][1]  - ranges[5][0]) + ranges[5][0]
    d_model     = int(round(sample[6]  * (ranges[6][1]  - ranges[6][0]) + ranges[6][0]))
    heads       = int(round(sample[7]  * (ranges[7][1]  - ranges[7][0]) + ranges[7][0]))
    d_ff        = int(round(sample[8]  * (ranges[8][1]  - ranges[8][0]) + ranges[8][0]))
    num_layers  = int(round(sample[9]  * (ranges[9][1]  - ranges[9][0]) + ranges[9][0]))
    kernel_size = int(round(sample[10] * (ranges[10][1] - ranges[10][0]) + ranges[10][0]))
    lstm_layers = int(round(sample[11] * (ranges[11][1] - ranges[11][0]) + ranges[11][0]))

    win_length = int(frame_s * sample_rate)
    hop_length = int(stride_s * sample_rate)
    # win_length <= n_fft
    if win_length > n_fft:
      print("FIXING: win_length <= n_fft")
      n_fft = int(frame_s * sample_rate)
    # kernel size must be odd number
    if kernel_size % 2 == 0:
      print("FIXING: kernel size must be odd number")
      kernel_size -= 1
    # hop_length <= win_length
    if hop_length > win_length:
      print("FIXING: hop_length <= win_length")
      stride_s = frame_s
    # d_model must be even number
    if d_model % 2 != 0:
      print("FIXING: d_model must be even number")
      d_model -= 1
    # d_model must be divisible by heads (as d_model is even, at least 2 heads)
    while d_model % heads != 0:
      heads -= 1

    values_abs =  np.array([
                    lr,
                    n_fft,
                    frame_s,
                    stride_s,
                    mse_weight,
                    snr_weight,
                    d_model,
                    heads,
                    d_ff,
                    num_layers,
                    kernel_size,
                    lstm_layers
                  ])
    pop = abs_to_pop(values_abs)
    
    np.clip(pop, a_min=0, a_max=None, out=pop)

    fixed.append(pop)

  population[:,:] = np.array(fixed)

def pop2dict(population): # population containing vector of ones and zeros
  
  sample_rate = 16000

  ranges = get_ranges()

  lr          = population[0]  * (ranges[0][1]  - ranges[0][0]) + ranges[0][0]
  n_fft       = int(round(population[1]  * (ranges[1][1]  - ranges[1][0]) + ranges[1][0]))
  frame_s     = population[2]  * (ranges[2][1]  - ranges[2][0]) + ranges[2][0]
  stride_s    = population[3]  * (ranges[3][1]  - ranges[3][0]) + ranges[3][0]
  mse_weight  = population[4]  * (ranges[4][1]  - ranges[4][0]) + ranges[4][0]
  snr_weight  = population[5]  * (ranges[5][1]  - ranges[5][0]) + ranges[5][0]
  d_model     = int(round(population[6]  * (ranges[6][1]  - ranges[6][0]) + ranges[6][0]))
  heads       = int(round(population[7]  * (ranges[7][1]  - ranges[7][0]) + ranges[7][0]))
  d_ff        = int(round(population[8]  * (ranges[8][1]  - ranges[8][0]) + ranges[8][0]))
  num_layers  = int(round(population[9]  * (ranges[9][1]  - ranges[9][0]) + ranges[9][0]))
  kernel_size = int(round(population[10] * (ranges[10][1] - ranges[10][0]) + ranges[10][0]))
  lstm_layers = int(round(population[11] * (ranges[11][1] - ranges[11][0]) + ranges[11][0]))

  ret = {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : lr,
    "momentum" : 0,
    "sample_rate" : sample_rate,
    "n_fft" : n_fft,
    "frame_s" : frame_s,
    "stride_s" : stride_s,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : mse_weight
                  },
    "loss_snr" : snr_weight,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : d_model,
                      "heads" : heads,
                      "d_ff" : d_ff,
                      "dropout" : 0.1,
                      "num_layers" : num_layers,
                      "depthwise_conv_kernel_size" : kernel_size,
                      "lstm_layers" : lstm_layers
                    }
  }

  return ret

def replace_max(array, index):
  ret = np.copy(array)
  ret[index] = 1
  return ret

def replace_min(array, index):
  ret = np.copy(array)
  ret[index] = 0
  return ret

def replace_middle(array, index):
  ret = np.copy(array)
  ret[index] = 0.5
  return ret

def get_initial_pop():
  
  initial = []
  init_abs = get_init_abs()

  init_pop = abs_to_pop(init_abs)
  ret = np.array([
    init_pop, # baseline
    #replace_max(init_pop, 0), # lr_max
    replace_min(init_pop, 0), # lr_min
    #replace_max(init_pop, 1), # n_fft_max
    replace_min(init_pop, 1), # n_fft_min
    #replace_max(init_pop, 2), # frame_s_max
    replace_min(init_pop, 2), # frame_s_min
    #replace_max(init_pop, 3), # stride_s_max
    replace_min(init_pop, 3), # stride_s_min
    #replace_max(init_pop, 4), # mse_weight_max
    replace_min(init_pop, 4), # mse_weight_min
    #replace_max(init_pop, 5), # snr_weight_max
    replace_min(init_pop, 5), # snr_weight_min
    #replace_max(init_pop, 6), # d_model_max
    replace_min(init_pop, 6), # d_model_min
    #replace_max(init_pop, 7), # heads_max
    replace_min(init_pop, 7), # heads_min
    #replace_max(init_pop, 8), # d_ff_max
    replace_min(init_pop, 8), # d_ff_min
    #replace_max(init_pop, 9), # num_layers_max
    replace_min(init_pop, 9), # num_layers_min
    #replace_max(init_pop, 10), # kernel_size_max
    replace_min(init_pop, 10), # kernel_size_min
    #replace_max(init_pop, 11), # lstm_layers_max
    replace_min(init_pop, 11),  # lstm_layers_min
    replace_middle(init_pop, 0), # lr_middle
    replace_middle(init_pop, 1), # n_fft_middle
    replace_middle(init_pop, 2), # frame_s_middle
    replace_middle(init_pop, 3), # stride_s_middle
    replace_middle(init_pop, 4), # mse_weight_middle
    replace_middle(init_pop, 5), # snr_weight_middle
    replace_middle(init_pop, 6), # d_model_middle
    replace_middle(init_pop, 7), # heads_middle
    replace_middle(init_pop, 8), # d_ff_middle
    replace_middle(init_pop, 9), # num_layers_middle
    replace_middle(init_pop, 10), # kernel_size_middle
    replace_middle(init_pop, 11), # lstm_layers_middle
  ])
  fix_pop(ret)

  print("Initial Population")
  pprint(ret)

  return ret