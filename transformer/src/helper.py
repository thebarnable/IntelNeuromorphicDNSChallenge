import numpy as np
from scipy.io import wavfile

def saveAudio(file_name, audio, toCPU=False, sample_rate=16000):
  if toCPU:
    my_audio = audio.cpu()
  else:
    my_audio = audio
  my_audio = np.array(my_audio.flatten(), dtype=np.float64)
  normalized_audio = my_audio / np.abs(my_audio).max()
  scaled_audio = np.int16((np.array(normalized_audio) * 32767))
  wavfile.write("../audio/" + file_name, sample_rate, scaled_audio)