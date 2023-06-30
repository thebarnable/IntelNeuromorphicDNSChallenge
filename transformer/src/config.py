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
# "network" : {
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
    "epochs" : 10,
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
  'example_vary' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 10,
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
                      "d_model" : 512,
                      "heads" : 8,
                      "enc" : 6,
                      "dec" : 6,
                      "d_ff" : 256,
                      "dropout" : 0.1
                    }
  },
  'test1' : {
    "device" : "cuda",
    "network" : "scaling",
    "epochs" : 10,
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
  }
}





