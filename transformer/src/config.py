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
  'testing' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 1,
    "batch" : 4,
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
  'scaling_2_SGD_e30_e50' : {
    "device" : "cuda",
    "network" : "scaling",
    "pretrained" : "scaling_2_SGD_e10_e30",
    "pretrained_epoch" : 30,
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





