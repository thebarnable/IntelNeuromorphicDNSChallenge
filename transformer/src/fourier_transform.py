import torch

class FourierTransform():
  def __init__(self, n_fft, stride, frame, polar=True):
    self.n_fft = n_fft
    self.stride = stride
    self.frame = frame
    self.polar = polar

  def stft(self, values):
    stft = torch.stft(values,
                      n_fft = self.n_fft,
                      hop_length=self.stride,
                      win_length=self.frame,
                      window=None,
                      center=True,
                      pad_mode='reflect',
                      normalized=True,
                      onesided=True,
                      return_complex=False
                    )
    if self.polar:
      real = stft[:, :, :, 0]
      imaginary = stft[:, :, :, 1]
      absolute = torch.sqrt(real ** 2 + imaginary ** 2)
      phase = torch.atan2(imaginary, real)
      return (absolute, phase)
    return stft

  def istft(self, values):
    if self.polar:
      magnitude = values[0]
      phase = values[1]
      real = magnitude * torch.cos(phase)
      imaginary = magnitude * torch.sin(phase)
      values = torch.stack((real, imaginary), dim=-1)
        
    istft = torch.istft(values,
                      n_fft = self.n_fft,
                      hop_length=self.stride,
                      win_length=self.frame,
                      window=None,
                      center=True,
                      normalized=True,
                      onesided=True,
                      return_complex=False
                    )
    return istft