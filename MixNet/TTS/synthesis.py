import torch
from TTS.api import TTS
from dnsmos import DNSMOS

tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)
wav = tts.tts("My name is Jason.", speaker_wav="C:/Users/nicol/Downloads/DNS/mySound.wav", language="en")
wav = tts.tts_to_file("My name is Jason.", speaker_wav="C:/Users/nicol/Downloads/DNS/mySound.wav", language="en",file_path="C:/Users/nicol/Downloads/DNS/myName.wav")
