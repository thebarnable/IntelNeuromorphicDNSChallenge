# Audio
num_mels = 64  # 80
warmup_steps = 16

# num_freq = 1024
n_fft = 512  # 1024
sr = 16000
hop_length = 160  # 160 # samples.
win_length = 400  # 400
embedding_size = 512
n_iter = 60

epochs = 10000
lr = 0.01
save_step = 16
image_step = 500
n_mfcc = 20
layers_DNN = 5
hidden_size_DNN = 2048
num_frames = 800
chunk_size = 20
num_chunks_per_process = 4 #4
gradient_accumulations = 1  # 5
weight_decay = 0.0
learning_rate_decay_interval = 10  # decay for every 100 epochs
learning_rate_decay_rate = 0.7  # lr = lr * rate #0.999
device = 'cpu'
cleaners = 'english_cleaners'
pad_value=0