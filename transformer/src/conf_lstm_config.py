###################
##### Options #####
###################
# "device" :       "cuda"/"cpu"
# "network" :      "generative"/"scaling"
# "epochs" :       integer > 0
# "batch" :        integer > 0
# "phase" :        True/False - whether phase or only magnitude is used
# "optimizer" :    "SGD"/"Adam"
# "lr" :           float > 0/"baseline"
# "momentum" :     float >= 0 - only relevant if optimizer is SGD
# "sample_rate" :  intger > 0
# "n_fft" :        integer, power of 2
# "frame_s" :   float > 0
# "stride_s" :  float > 0    
# "loss_mse" :     {
#                    "mode" : "scale"/"frequency", - only relevant if network is scaling
#                    "weight" : float >= 0 - weight to consider mse in loss,
#                  }
# "loss_snr" :     float >= 0 - weight to consider snr in loss

############################################################################
#### Network parameters from Attention is all you need (Vaswani et al.) ####
############################################################################
# "transformer" : {
#               "d_model" : 512,
#               "heads" : 8,
#               "enc" : 6,
#               "dec" : 6,
#               "d_ff" : 256,
#               "dropout" : 0.1
#             }

config = {
  'example' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 1,
    "phase" : False,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'test1' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 1,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  # genetic init 1000 found:
  'genetic_init_1000' : {'batch': 2,
 'device': 'cuda',
 'epochs': 1,
 'frame_s': 0.02,
 'loss_mse': {'mode': 'scale', 'weight': 1.0},
 'loss_snr': 1.0,
 'lr': 0.02,
 'momentum': 0,
 'n_fft': 816,
 'network': 'scaling',
 'optimizer': 'SGD',
 'phase': True,
 'sample_rate': 16000,
 'stride_s': 0.01,
 'transformer': {'conformer': True,
                 'd_ff': 143,
                 'd_model': 32,
                 'depthwise_conv_kernel_size': 31,
                 'dropout': 0.1,
                 'heads': 2,
                 'lstm_layers': 1,
                 'num_layers': 12}},

  'genetic_init_1000_lr02' : {'batch': 2, # wrong lr!!!
 'device': 'cuda',
 'epochs': 1,
  "pretrained" : "genetic_init_1000",
  "pretrained_epoch" : 1,
 'frame_s': 0.02,
 'loss_mse': {'mode': 'scale', 'weight': 1.0},
 'loss_snr': 1.0,
 'lr': 0.08895284892334082,
 'momentum': 0,
 'n_fft': 816,
 'network': 'scaling',
 'optimizer': 'SGD',
 'phase': True,
 'sample_rate': 16000,
 'stride_s': 0.01,
 'transformer': {'conformer': True,
                 'd_ff': 143,
                 'd_model': 32,
                 'depthwise_conv_kernel_size': 31,
                 'dropout': 0.1,
                 'heads': 2,
                 'lstm_layers': 1,
                 'num_layers': 12}},

  'genetic_init_1000_lr02_e2_5' : {'batch': 2,
 'device': 'cuda',
 'epochs': 3,
  "pretrained" : "genetic_init_1000_lr02",
  "pretrained_epoch" : 2,
 'frame_s': 0.01,
 'loss_mse': {'mode': 'scale', 'weight': 1.0},
 'loss_snr': 1.0,
 'lr': 0.02,
 'momentum': 0,
 'n_fft': 816,
 'network': 'scaling',
 'optimizer': 'SGD',
 'phase': True,
 'sample_rate': 16000,
 'stride_s': 0.01,
 'transformer': {'conformer': True,
                 'd_ff': 143,
                 'd_model': 32,
                 'depthwise_conv_kernel_size': 31,
                 'dropout': 0.1,
                 'heads': 2,
                 'lstm_layers': 1,
                 'num_layers': 12}},


  # NEW
  'conformer_lstm_lr05_snr02_lr01' : {
    "device" : "cuda",
    "network" : "scaling",
    "pretrained" : "conformer_lstm_lr05_snr02",
    "pretrained_epoch" : 2,
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.2,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_lr05_snr02_contd_e4' : {
    "device" : "cuda",
    "network" : "scaling",
    "pretrained" : "conformer_lstm_lr05_snr02",
    "pretrained_epoch" : 2,
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.2,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_lr05_0_03' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 3,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 0
                  },
    "loss_snr" : 0.3,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  # validation
  'val_conformer_lstm_lr05_snr02' : {
    "device" : "cuda",
    "network" : "scaling",
    "validation" : True,
    "pretrained" : "conformer_lstm_lr05_snr02",
    "pretrained_epoch" : 2,
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.2,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'val_conformer_lstm_lr05' : {
    "device" : "cuda",
    "network" : "scaling",
    "validation" : True,
    "pretrained" : "conformer_lstm_lr05",
    "pretrained_epoch" : 2,
    "epochs" : 2,
    "batch" : 1,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },



  # conformer_lstm_lr05_...
  'conformer_lstm_lr05_e2' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_lr1' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.1,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_lr05_snr02' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.2,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_lr_05_mse05' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 0.5
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },

  # conformer_lstm_l6_mse_freq
  'conformer_lstm_l6_mse_freq' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_mse_freq_lr05' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_mse_freq_snr02' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0.2,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_mse_freq_mse05' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 0.5
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },

  # conf_lstm new exploration

  # conformer_lstm
  'conformer_lstm' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l4_h4' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l4_kernel_15' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_lstm_l4_h4_kernel_15' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_lstm_l4_h4_kernel_15_dff128' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 128,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_lstm_l4_h2_kernel_15_dff128' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 2,
                      "d_ff" : 128,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_lstm_l4_h4_kernel_7_dff128' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 128,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 7
                    }
  },
  'conformer_lstm_l4_h4_kernel_7_dff64' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 64,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 7
                    }
  },
  # conf_lstm explo
  'conformer_lstm_l6_no_phase' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : False,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_Adam' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_mse_freq' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_kernel_15' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_lstm_l6_h4' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_gen' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_gen_Adam' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_gen_no_phase' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 2,
    "batch" : 3,
    "phase" : False,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6_gen_lr02' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.02,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  # conformer_lstm layers
  'conformer_lstm_l4' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l6' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l8' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 8,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_l10' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 10,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  # conf_lstm loss
  'conformer_lstm_1_01' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_lr05' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_05_01' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 0.5
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_1_02' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.2,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_1_03' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.3,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lstm_1_1' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },

  ######################### NEW END





  #### conf_lstm_end
  # conformer
  'conformer_l4_h4' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l4_kernel_15' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_l4_h4_kernel_15' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_l4_h4_kernel_15_dff128' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 128,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_l4_h2_kernel_15_dff128' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 2,
                      "d_ff" : 128,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_l4_h4_kernel_7_dff128' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 128,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 7
                    }
  },
  'conformer_l4_h4_kernel_7_dff64' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 64,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 7
                    }
  },
  # conf explo
  'conformer_l6_no_phase' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : False,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l6_Adam' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l6_mse_freq' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l6_kernel_15' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 15
                    }
  },
  'conformer_l6_h4' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l6_gen' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l6_gen_Adam' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l6_gen_no_phase' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 2,
    "batch" : 3,
    "phase" : False,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l6_gen_lr02' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 2,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.02,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l6_gen_e10' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 10,
    "batch" : 3,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  # conformer layers
  'conformer_l4' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 4,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l6' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 6,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l8' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 8,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_l10' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 2,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 10,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  # conf loss
  'conformer_1_01' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_lr05' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 1,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.05,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_05_01' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 0.5
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_1_02' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.2,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_1_03' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.3,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },
  'conformer_1_1' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 1,
    "transformer" : {
                      "conformer" : True,
                      "d_model" : 256,
                      "heads" : 8,
                      "d_ff" : 256,
                      "dropout" : 0.1,
                      "num_layers" : 12,
                      "depthwise_conv_kernel_size" : 31
                    }
  },



  # continue training
  'scaling_2_SGD_lr0_1' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.1,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_SGD_no_phase' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : False,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_SGD_e10_e30' : {
    "device" : "cuda",
    "network" : "scaling",
    "pretrained" : "scaling_2_SGD_e10",
    "pretrained_epoch" : 10,
    "epochs" : 20,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },

  # 10 epochs
  'scaling_2_SGD_mse0_5_e10' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 10,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 0.5
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_SGD_e10' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 10,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_e10' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 10,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },

  # Exploration STFT
  'scaling_2_SGD_fft_1024' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 1024,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_SGD_fft_256' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 256,
    "frame_s" : 0.016,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_SGD_fft_1024_05_02' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 1024,
    "frame_s" : 0.05,
    "stride_s" : 0.02,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },

  # Exploration - scaling_2_SGD
  'scaling_2_SGD_large' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 1,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                  "d_model" : 512,
                  "heads" : 8,
                  "enc" : 6,
                  "dec" : 6,
                  "d_ff" : 256,
                  "dropout" : 0.1
                }
  },
  'scaling_2_SGD_snr0_2' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.2,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_SGD_mse0_5' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 0.5
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_SGD_e3' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 3,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_SGD_large_e3' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 3,
    "batch" : 1,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                  "d_model" : 512,
                  "heads" : 8,
                  "enc" : 6,
                  "dec" : 6,
                  "d_ff" : 256,
                  "dropout" : 0.1
                }
  },



  # Exploration - baseline

  # scaling
  'scaling_0' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_1_frequency_loss' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_2_SGD' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_3_no_phase' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : False,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  # generative
  'generative_0' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'generative_2_SGD' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'generative_3_no_phase' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 1,
    "batch" : 2,
    "phase" : False,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0.1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  # scaling - mse only
  'scaling_mse_0' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_mse_1_frequency_loss' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_mse_2_SGD' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_mse_3_no_phase' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : False,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 1
                  },
    "loss_snr" : 0,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  # generative
  'generative_mse_0' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'generative_mse_2_SGD' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'generative_mse_3_no_phase' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 1,
    "batch" : 2,
    "phase" : False,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 1
                  },
    "loss_snr" : 0,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  # scaling - snr only
  'scaling_snr_0' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 0
                  },
    "loss_snr" : 1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_snr_1_frequency_loss' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 0
                  },
    "loss_snr" : 1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_snr_2_SGD' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 0
                  },
    "loss_snr" : 1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'scaling_snr_3_no_phase' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 2,
    "phase" : False,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "scale",
                    "weight" : 0
                  },
    "loss_snr" : 1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  # generative
  'generative_snr_0' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 0
                  },
    "loss_snr" : 1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'generative_snr_2_SGD' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 1,
    "batch" : 2,
    "phase" : True,
    "optimizer" : "SGD",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 0
                  },
    "loss_snr" : 1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'generative_snr_3_no_phase' : {
    "device" : "cuda",
    "network" : "generative",
    "epochs" : 1,
    "batch" : 2,
    "phase" : False,
    "optimizer" : "Adam",
    "lr" : 0.01,
    "momentum" : 0,
    "sample_rate" : 16000,
    "n_fft" : 512,
    "frame_s" : 0.025,
    "stride_s" : 0.01,
    "loss_mse" :  {
                    "mode" : "frequency",
                    "weight" : 0
                  },
    "loss_snr" : 1,
    "transformer" : {
                      "d_model" : 256,
                      "heads" : 8,
                      "enc" : 4,
                      "dec" : 4,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  }
}





