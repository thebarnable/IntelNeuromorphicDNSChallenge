import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms

class STFT(nn.Module):
  def stft(self, input):
    return torch.stft(input,
                      n_fft = self.n_fft,
                      hop_length=self.hop_length,
                      win_length=self.frame,
                      window=None,
                      center=True,
                      pad_mode='reflect',
                      normalized=True,
                      onesided=True,
                      return_complex=False
                    )
            
  def istft(self, input):
    return torch.istft(input,
                      n_fft = self.n_fft,
                      hop_length=self.hop_length,
                      win_length=self.frame,
                      window=None,
                      center=True,
                      normalized=True,
                      onesided=True,
                      return_complex=False
                    )

  def __init__(self):
    super(STFT, self).__init__()
    self.sample_rate = 16000
    self.n_fft = 512
    self.frame = int(self.sample_rate*0.025)
    self.hop_length = int(self.sample_rate*0.01)


  def forward(self, waveform):
    print(waveform.shape)
    spec = self.stft(waveform)
    print(spec.shape)
    time = self.istft(spec)
    print(time.shape)
    return time


class GriffinLim(nn.Module):
  def __init__(self):
    super(GriffinLim, self).__init__()

    sample_rate = 16000
    self.frame = int(sample_rate*0.025)
    self.hop_length = int(sample_rate*0.01)

    self.spectrogram = transforms.Spectrogram(
      n_fft=self.frame,
      hop_length=self.hop_length,
      center=True,
      pad_mode='reflect',
      power=None
    )

    self.inverse_spectrogram = transforms.GriffinLim(
      n_fft=self.frame,
      hop_length=self.hop_length
    )

  def forward(self, waveform):
    print("Forward")
    print(waveform.shape)
    spectrogram = self.spectrogram(waveform)
    print(spectrogram.shape)
    spectrogram = torch.abs(spectrogram)
    print(spectrogram.shape)
    reconstructed_waveform = self.inverse_spectrogram(spectrogram)
    print(reconstructed_waveform.shape)
    
    return reconstructed_waveform


class NetWaveGlow(nn.Module):
  def __init__(self, device):
    super(NetWaveGlow, self).__init__()

    sample_rate = 16000 # TODO 20050 is used for pre-trained waveglow model
    self.frame = int(sample_rate*0.025)
    self.hop_length = int(sample_rate*0.01)

    #self.mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=80, dct_type=2, norm='ortho', log_mels=True, melkwargs={"n_fft": self.frame, "hop_length": self.hop_length, "n_mels": 80, "center": True})

    self.mfcc = torchaudio.transforms.MFCC(
    sample_rate=sample_rate,
    n_mfcc=80,
    melkwargs={
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 80,
        "center": False,
        "norm": None,
        "power": 2.0,
        "pad_mode": "reflect",
        "window_fn": torch.hann_window
      }
    )

    # Workaround to load model mapped on GPU
    # https://stackoverflow.com/a/61840832
    self.waveglow = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub",
        "nvidia_waveglow",
        model_math="fp32",
        pretrained=False,
    )
    checkpoint = torch.load("waveglow/waveglow_checkpoint.pth", map_location=device)
    state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

    self.waveglow.load_state_dict(state_dict)
    self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)


  def forward(self, waveform):
    print("Forward")
    print(waveform.shape)
    mfcc = self.mfcc(waveform)
    print(mfcc.shape)
    waveforms = self.waveglow.infer(mfcc)
    print(waveforms.shape)
    
    return waveforms

