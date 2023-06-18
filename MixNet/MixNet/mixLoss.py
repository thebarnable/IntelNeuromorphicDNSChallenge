import librosa
import matplotlib
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

import hyperparams as hp
import torchaudio.transforms as T
import numpy as np
from typing import Union

import create_chunks
from temp.compSparsity import compSparsity


class MixLossSlice(nn.Module):
    """
    This object is the Loss which is optimized by MixNet
    """

    def __init__(self):
        super(MixLossSlice, self).__init__()

    def forward(self, output, target, mel_noise, mel_tts):
        criterion = nn.MSELoss()
        loss = criterion(output, target)

        reg_frame = 0
        output_split = output.split(1, dim=2)
        tts_split = mel_tts.split(1, dim=2)
        noise_split = mel_noise.split(1, dim=2)
        for idx, frame in enumerate(output_split):
            reg_frame = reg_frame + criterion(output_split[idx],noise_split[idx])#* criterion(tts_split[idx], noise_split[idx])
        return loss + 0.5 * reg_frame


class MixLoss(nn.Module):
    """
    This object is the Loss which is optimized by MixNet
    """

    def __init__(self):
        super(MixLoss, self).__init__()

    def forward(self, output, target, mel_noise, mel_tts):
        criterion = nn.MSELoss()
        loss = criterion(output, target)

        reg = criterion(output, mel_tts) #* criterion(mel_tts, mel_noise)
        reg_chunk = 0
        output_split = output.split(hp.chunk_size, dim=2)
        tts_split = mel_tts.split(hp.chunk_size, dim=2)
        noise_split = mel_noise.split(hp.chunk_size, dim=2)

        for idx, chunk in enumerate(output_split):
            reg_chunk = reg_chunk + criterion(output_split[idx], noise_split[idx]) * criterion(tts_split[idx], noise_split[idx])

        return 0.5*loss + reg_chunk + 2*reg


def mel_si_snr(target: Union[torch.tensor, np.ndarray],
               estimate: Union[torch.tensor, np.ndarray]) -> torch.tensor:
    """Calculates SI-SNR estiamte from target mel and estiamte mel. The
    mel sequene is expected to be a tensor/array of dimension more than 1.
    The last dimension is interpreted as time.
    The implementation is based on the example here:
    https://www.tutorialexample.com/wp-content/uploads/2021/12/SI-SNR-definition.png
    Parameters
    ----------
    target : Union[torch.tensor, np.ndarray]
        Target audio melspec.
    estimate : Union[torch.tensor, np.ndarray]
        Estimate audio melspec.
    Returns
    -------
    torch.tensor
        SI-SNR of each target and estimate pair.
    """
    EPS = 1e-8

    if not torch.is_tensor(target):
        target: torch.tensor = torch.tensor(target)
    if not torch.is_tensor(estimate):
        estimate: torch.tensor = torch.tensor(estimate)

    # zero mean to ensure scale invariance
    s_target = target - torch.mean(target, dim=0, keepdim=True)
    s_estimate = estimate - torch.mean(estimate, dim=0, keepdim=True)
    s_target = torch.flatten(s_target)
    s_estimate = torch.flatten(s_estimate)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_norm = torch.sum(s_target ** 2, dim=-1, keepdim=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = torch.sum(pair_wise_proj ** 2,
                              dim=-1) / (torch.sum(e_noise ** 2,
                                                   dim=-1) + EPS)
    return 10 * torch.log10(pair_wise_sdr + EPS)


def si_snr(target: Union[torch.tensor, np.ndarray],
           estimate: Union[torch.tensor, np.ndarray]) -> torch.tensor:
    """Calculates SI-SNR estiamte from target audio and estiamte audio. The
    audio sequene is expected to be a tensor/array of dimension more than 1.
    The last dimension is interpreted as time.
    The implementation is based on the example here:
    https://www.tutorialexample.com/wp-content/uploads/2021/12/SI-SNR-definition.png
    Parameters
    ----------
    target : Union[torch.tensor, np.ndarray]
        Target audio waveform.
    estimate : Union[torch.tensor, np.ndarray]
        Estimate audio waveform.
    Returns
    -------
    torch.tensor
        SI-SNR of each target and estimate pair.
    """
    EPS = 1e-8

    if not torch.is_tensor(target):
        target: torch.tensor = torch.tensor(target)
    if not torch.is_tensor(estimate):
        estimate: torch.tensor = torch.tensor(estimate)

    # zero mean to ensure scale invariance
    s_target = target - torch.mean(target, dim=-1, keepdim=True)
    s_estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_norm = torch.sum(s_target ** 2, dim=-1, keepdim=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = torch.sum(pair_wise_proj ** 2,
                              dim=-1) / (torch.sum(e_noise ** 2,
                                                   dim=-1) + EPS)
    return 10 * torch.log10(pair_wise_sdr + EPS)


def strech_signal(reference, input):
    """
    This function stretches the waveform input to the length of the reference waveform
    :param reference: numpy array of shape [N,]
    :param input: numpy array of shape [M,]
    :return: numpy array of shape [N,]
    """
    ref_length = reference.shape[0]
    input_length = input.shape[0]
    factor = input_length / ref_length
    out = librosa.effects.time_stretch(y=input, rate=factor)
    return out


def plot_mel(mel, mel_pred, mel_target):
    """
    This function plots the three mel spectrograms of shape [1,features,frames]
    mel: the imput mel spectrogram with both synthetic and noisy speech joined together (has 2*num_frames size)
    mel_pred: The predicted mel spec from the model
    mel_target: The target mel
    """
    fig, axs = plt.subplots(3)
    axs[0].set_title('Mel_Input')
    axs[0].set_ylabel('mel freq')
    axs[0].set_xlabel('frame')
    im = axs[0].imshow(librosa.power_to_db(mel[0, :, :].cpu().squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[0])
    axs[1].set_title('Mel_pred')
    axs[1].set_ylabel('mel freq')
    axs[1].set_xlabel('frame')
    im = axs[1].imshow(librosa.power_to_db(mel_pred.detach()[0, :, :].cpu().squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[1])
    axs[2].set_title('Mel_clean')
    axs[2].set_ylabel('mel freq')
    axs[2].set_xlabel('frame')
    im = axs[2].imshow(librosa.power_to_db(mel_target.detach()[0, :, :].cpu().squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[2])
    fig.tight_layout(pad=0.5)
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.savefig('melCh1.svg')
    matplotlib.pyplot.close()
