import matplotlib
from matplotlib import pyplot as plt

import chunkGenNet

plt.rcParams['axes.grid'] = True
import random
import shiftMel

from TTS.api import TTS
from tqdm import tqdm
import mixLoss

import torch as t
import torchaudio
import torchaudio.transforms as T

import librosa
from typing import Optional

from IntelNeuromorphicDNSChallenge.MixNet.DataLoader.data import get_batch_loader
from IntelNeuromorphicDNSChallenge.MixNet.DataLoader.data_loaders import TextLoader
from IntelNeuromorphicDNSChallenge.MixNet.DataLoader.padder import get_padders
from IntelNeuromorphicDNSChallenge.MixNet.DataLoader.pipelines import get_pipelines
from IntelNeuromorphicDNSChallenge.MixNet.DataLoader.tokenizer import CharTokenizer
import torch.nn.functional as F
import torchaudio.functional

import numpy as np
import hyperparams as hp
from torchaudio.utils import download_asset

from IntelNeuromorphicDNSChallenge.MixNet.DataLoader.args import (
    get_args,
    get_aud_args,
    get_data_args,
)

import os
import torch


def get_tokenizer(args):
    """
    This function gives the tokenizer for the dataloader. It can be used together with MultiSpeech TTS
    """
    tokenizer = CharTokenizer()
    tokenizer_path = args.tokenizer_path
    if args.tokenizer_path is not None:
        tokenizer.load_tokenizer(tokenizer_path)
        return tokenizer
    data = TextLoader(args.train_path).load().split('\n')
    data = list(map(lambda x: (x.split(args.sep))[2], data))

    tokenizer.add_pad_token().add_eos_token()
    tokenizer.set_tokenizer(data)
    tokenizer_path = os.path.join(args.checkpoint_dir, 'tokenizer.json')
    tokenizer.save_tokenizer(tokenizer_path)
    print(f'tokenizer saved to {tokenizer_path}')
    return tokenizer

def batch_mels(data, mel_spectrogram, num_frames, tts):
    """
    This function creates the datset. It simulates 10% WER and returns the mel spectrogram the clean target
    audio and the along the time axis stacked and padded tts and noisy audio. It aligns TTS and noisy with the strech_wavs
    function. Last, it batches the data.
    data: embedding_path,file_path,text
    mel_spectrogram: The mel spec transformation
    num_frames: The # of frames one single mel spec (noisy, tts, clean) will be padded to (800)
    tts: The TTS model
    """

    mels_tts = []
    mels_noise = []
    mels_clean = []
    embeds_paths, file_paths, texts = data
    # dictionary for similar sounding words
    dictionary = {'DRESS': ['MESS', 'REST', 'GUESS'], 'MIND': ['KIND', 'WIND'], 'STAIRS': ['CARES', 'SQUARE']}
    # dictionary for random words
    dictionary_rand = ['DRESS', 'MESS', 'REST', 'GUESS''MIND', 'KIND', 'WIND' 'STAIRS', 'CARES', 'SQUARE']

    for file_path, embeds_path, text in zip(file_paths, embeds_paths, texts):

        wav, in_sr = librosa.load(file_path)
        wav = torch.from_numpy(wav).float()
        wav = torchaudio.functional.resample(wav, in_sr, hp.sr, lowpass_filter_width=6)

        SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
        noise, _ = torchaudio.load(SAMPLE_NOISE)
        noise = torch.cat((
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise), 1)

        noise = torchaudio.functional.resample(noise, 8000, hp.sr, lowpass_filter_width=6)
        noise_add = noise[:, : wav.shape[0]]
        snr_dbs = torch.tensor([0])
        waveform_noise = add_noise(waveform=wav.unsqueeze(0), noise=noise_add, snr=snr_dbs)

        ####################TTS begin###############

        words = text.split()
        random.randint(0, len(dictionary) - 1)
        for j in range(len(words)):
            if random.randint(0, 9) == 9:
                replacement = dictionary_rand[random.randrange(len(dictionary_rand))]
                words[j] = replacement
        text = ' '.join(words)

        """
        words = text.split()
        for j in range(len(words)):
            word = words[j]
            replacement = dictionary.get(word)
            if replacement:
                words[j] = random.choice(replacement)
        text = ' '.join(words)
        """

        wav_tts = tts.tts(text, speaker_wav=embeds_path,
                          language="en")
        wav_tts = torch.FloatTensor(wav_tts)

        ##################TTS end##################

        shape_model = waveform_noise.squeeze(0).numpy()
        # ALIGN THE WAVEFORMS
        wav_tts = torch.from_numpy(strech_signal(shape_model, wav_tts.numpy().astype(float)).astype(float))

        mels_clean.append(mel_spectrogram(wav.unsqueeze(0)))
        # wavs_noise.append(wav.unsqueeze(0))
        mels_noise.append(mel_spectrogram(waveform_noise))
        mels_tts.append(mel_spectrogram(wav_tts.unsqueeze(0).float()))
        # wavs_tts.append(wav.unsqueeze(0))

    max_len = 0
    for p, val in enumerate(mels_tts):
        length = val.size(2)
        if (length > max_len):
            max_len = length
    for p, val in enumerate(mels_tts):
        mels_tts[p] = F.pad(val, (
            max_len - val.size(2), 0))

    max_len = 0
    for p, val in enumerate(mels_noise):
        length = val.size(2)
        if (length > max_len):
            max_len = length
    for p, val in enumerate(mels_noise):
        mels_noise[p] = F.pad(val, (
            max_len - val.size(2), 0))

    max_len = 0
    for p, val in enumerate(mels_clean):
        length = val.size(2)
        if (length > max_len):
            max_len = length
    for p, val in enumerate(mels_clean):
        mels_clean[p] = F.pad(val, (
            max_len - val.size(2), 0))

    melspec_target = torch.cat(mels_clean, dim=0)
    melspec_noise = torch.cat(mels_noise, dim=0)
    melspec_tts = torch.cat(mels_tts, dim=0)

    pad_len_noise = melspec_noise.size(2)
    pad_len_tts = melspec_tts.size(2)
    mel_appnd = hp.pad_value*torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_tts).to(hp.device)
    melspec_tts = (torch.cat((mel_appnd, melspec_tts), dim=2))
    mel_appnd = hp.pad_value*torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_noise).to(hp.device)
    melspec_noise = (torch.cat((mel_appnd, melspec_noise), dim=2))

    melspec_tts= shiftMel.shiftMel(melspec_tts.unsqueeze(1), 2, 20).squeeze(1)

    mel = torch.cat((melspec_tts.unsqueeze(1), melspec_noise.unsqueeze(1)), 1)

    pad_len_clean = melspec_target.size(2)
    mel_appnd = hp.pad_value*torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_clean).to(hp.device)
    mel_target = (torch.cat((mel_appnd, melspec_target), dim=2))

    mel = torch.permute(mel, (0, 1, 3, 2))  # [B,C,T,F]
    mel_target = torch.permute(mel_target, (0, 2, 1))  # [B,T,F]

    return mel, mel_target


def batch_mels_wav(data, mel_spectrogram, num_frames, tts):
    """
    This function creates the datset. It simulates 10% WER and returns the mel spectrogram the clean target
    audio and the along the time axis stacked and padded tts and noisy audio. It aligns TTS and noisy with the strech_wavs
    function. Last, it batches the data.
    data: embedding_path,file_path,text
    mel_spectrogram: The mel spec transformation
    num_frames: The # of frames one single mel spec (noisy, tts, clean) will be padded to (800)
    tts: The TTS model
    """

    wavs_tts = []
    wavs_noise = []
    wavs_clean = []
    embeds_paths, file_paths, texts = data
    # dictionary for similar sounding words
    dictionary = {'DRESS': ['MESS', 'REST', 'GUESS'], 'MIND': ['KIND', 'WIND'], 'STAIRS': ['CARES', 'SQUARE']}
    # dictionary for random words
    dictionary_rand = ['DRESS', 'MESS', 'REST', 'GUESS''MIND', 'KIND', 'WIND' 'STAIRS', 'CARES', 'SQUARE']

    for file_path, embeds_path, text in zip(file_paths, embeds_paths, texts):

        wav, in_sr = librosa.load(file_path)
        wav = torch.from_numpy(wav).float()
        wav = torchaudio.functional.resample(wav, in_sr, hp.sr, lowpass_filter_width=6)

        SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
        noise, _ = torchaudio.load(SAMPLE_NOISE)
        noise = torch.cat((
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise, noise,
            noise, noise, noise, noise, noise, noise, noise), 1)

        noise = torchaudio.functional.resample(noise, 8000, hp.sr, lowpass_filter_width=6)
        noise_add = noise[:, : wav.shape[0]]
        snr_dbs = torch.tensor([0])
        waveform_noise = add_noise(waveform=wav.unsqueeze(0), noise=noise_add, snr=snr_dbs)

        ####################TTS begin###############

        words = text.split()
        random.randint(0, len(dictionary) - 1)
        for j in range(len(words)):
            if random.randint(0, 9) == 9:
                replacement = dictionary_rand[random.randrange(len(dictionary_rand))]
                words[j] = replacement
        text = ' '.join(words)

        """
        words = text.split()
        for j in range(len(words)):
            word = words[j]
            replacement = dictionary.get(word)
            if replacement:
                words[j] = random.choice(replacement)
        text = ' '.join(words)
        """

        wav_tts = tts.tts(text, speaker_wav=embeds_path,
                          language="en")
        wav_tts = torch.FloatTensor(wav_tts)

        ##################TTS end##################

        shape_model = waveform_noise.squeeze(0).numpy()
        # ALIGN THE WAVEFORMS
        wav_tts = torch.from_numpy(strech_signal(shape_model, wav_tts.numpy().astype(float)).astype(float))

        wavs_clean.append(wav.unsqueeze(0))
        # wavs_noise.append(wav.unsqueeze(0))
        wavs_noise.append(waveform_noise)
        wavs_tts.append(wav_tts.unsqueeze(0).float())
        # wavs_tts.append(wav.unsqueeze(0))

    max_len = 0
    for p, val in enumerate(wavs_tts):
        length = val.size(1)
        if (length > max_len):
            max_len = length
    for p, val in enumerate(wavs_tts):
        wavs_tts[p] = F.pad(val, (
            max_len - val.size(1), 0))

    max_len = 0
    for p, val in enumerate(wavs_noise):
        length = val.size(1)
        if (length > max_len):
            max_len = length
    for p, val in enumerate(wavs_noise):
        wavs_noise[p] = F.pad(val, (
            max_len - val.size(1), 0))

    max_len = 0
    for p, val in enumerate(wavs_clean):
        length = val.size(1)
        if (length > max_len):
            max_len = length
    for p, val in enumerate(wavs_clean):
        wavs_clean[p] = F.pad(val, (
            max_len - val.size(1), 0))

    wavs_clean = torch.cat(wavs_clean, dim=0)
    wavs_noise = torch.cat(wavs_noise, dim=0)
    wavs_tts = torch.cat(wavs_tts, dim=0)

    melspec_target = mel_spectrogram(wavs_clean.to(hp.device))
    melspec_noise = mel_spectrogram(wavs_noise.to(hp.device))
    melspec_tts = mel_spectrogram(wavs_tts.to(hp.device))

    pad_len_noise = melspec_noise.size(2)
    pad_len_tts = melspec_tts.size(2)
    mel_appnd = hp.pad_value*torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_tts).to(hp.device)
    melspec_tts = (torch.cat((mel_appnd, melspec_tts), dim=2))
    mel_appnd = hp.pad_value*torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_noise).to(hp.device)
    melspec_noise = (torch.cat((mel_appnd, melspec_noise), dim=2))

    melspec_tts= shiftMel.shiftMel(melspec_tts.unsqueeze(1), 2, 20).squeeze(1)

    mel = torch.cat((melspec_tts.unsqueeze(1), melspec_noise.unsqueeze(1)), 1)

    pad_len_clean = melspec_target.size(2)
    mel_appnd = hp.pad_value*torch.ones(melspec_noise.size(0), melspec_noise.size(1), num_frames - pad_len_clean).to(hp.device)
    mel_target = (torch.cat((mel_appnd, melspec_target), dim=2))

    mel = torch.permute(mel, (0, 1, 3, 2))  # [B,C,T,F]
    mel_target = torch.permute(mel_target, (0, 2, 1))  # [B,T,F]
    return mel, mel_target


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    args = get_args()

    tokenizer = get_tokenizer(args)
    aud_args = get_aud_args(args)
    data_args = get_data_args(args)

    text_padder, aud_padder = get_padders(0, tokenizer.special_tokens.pad_id)
    audio_pipeline, text_pipeline = get_pipelines(tokenizer, aud_args)

    data_loader = get_batch_loader(
        TextLoader(args.train_path),
        audio_pipeline,
        text_pipeline,
        aud_padder,
        text_padder,
        **data_args
    )

    test_loader = get_batch_loader(
        TextLoader(args.test_path),
        audio_pipeline,
        text_pipeline,
        aud_padder,
        text_padder,
        **data_args
    )

    mel_spectrogram = T.MelSpectrogram(
        sample_rate=hp.sr,
        n_fft=hp.n_fft,
        win_length=hp.win_length,
        hop_length=hp.hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=hp.num_mels,
        mel_scale="htk",
    ).to(hp.device)

    train_losses = []
    test_losses = []
    gradients = []
    tmp = []

    # TTS
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=False)

    # load model+hyperparameters
    curr_lr = hp.lr
    num_frames = hp.num_frames
    num_chunks_per_process = hp.num_chunks_per_process
    model = chunkGenNet.ChunkGenNet(num_chunks_per_process * hp.chunk_size * 2 * hp.num_mels, hp.layers_DNN,
                                    hp.hidden_size_DNN, hp.num_mels * hp.chunk_size,
                                    num_chunks_per_process).to(
        hp.device)
    print("This model has " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)) + " parameters")
    # loss
    loss_function = mixLoss.MixLoss()
    # optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=curr_lr, momentum=0.9, weight_decay=hp.weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=curr_lr, weight_decay=hp.weight_decay)
    optimizer.zero_grad()

    epoch_start = 0
    if epoch_start > 0:
        state_dict = t.load('./models/checkpoint_%s_%d.pth.tar' % ("MixNet", epoch_start))
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])

    global_step = 0
    for epoch in range(hp.epochs):
        # the accumulated training and testing losses over one epoch
        if epoch > 0:
            train_losses.append(train_loss)
            test_losses.append(test_loss)
        pbar = tqdm(data_loader)
        train_loss = 0
        test_loss = 0
        # for every batch
        for i, data in enumerate(data_loader):
            pbar.set_description("Processing at epoch %d" % epoch)
            global_step += 1
            if global_step < hp.warmup_steps:
                curr_lr = hp.lr * global_step / hp.warmup_steps
                update_lr(optimizer, curr_lr)
            elif epoch % hp.learning_rate_decay_interval == 0:
                curr_lr *= hp.learning_rate_decay_rate  # lr = lr * rate
                update_lr(optimizer, curr_lr)

            # extract data of shape [B,frames,features]
            mel, mel_target = batch_mels_wav(data=data, mel_spectrogram=mel_spectrogram,
                                             num_frames=num_frames,
                                             tts=tts)

            mel = mel.to(hp.device)
            mel_target = mel_target.to(hp.device)

            mel_pred = model(mel)  # forward pass # [B,F,T]

            # only consider the non-padded interval of the spec: padded columns are automatically mapped to zero
            mel_shift = mel[:, 1, :, :] + (-hp.pad_value)*torch.ones_like(mel[:, 1, :, :])
            non_empty_mask = mel_shift.abs().sum(dim=2).bool()
            mel_pred = torch.permute(mel_pred, (0, 2, 1))
            mel_pred[~non_empty_mask, :] = hp.pad_value
            mel_pred = torch.permute(mel_pred, (0, 2, 1))
            mel_target = mel_target.permute(0, 2, 1)
            mel = mel.permute(0, 1, 3, 2)
            # plot results
            plot_mel(mel, mel_pred, mel_target)
            # compute the loss
            loss = loss_function(mel_pred, mel_target, mel[:, 1, :, :], mel[:, 0, :, :])
            loss.requires_grad_(True)

            # run backpass without gradient accumulation: equal to hp.gradient_accumulations=1
            # loss.backward()
            # run backpass
            if i % hp.gradient_accumulations == 0:
                # Accumulates gradient before each step
                loss.backward()
                for param in model.parameters():
                    if not param.grad == None:
                        tmp.append(param.grad.abs().cpu().mean())
                gradients.append(np.mean(tmp))  # plot gradients
                tmp = []
                optimizer.step()
                optimizer.zero_grad()
            else:
                loss.backward(retain_graph=True)  # error

            # update weights
            # optimizer.step()

            # for tracing the loss
            train_loss = train_loss + loss.item()

            # testing
            # test_loss = test_loss + test(model, test_loader, mel_spectrogram, num_frames, loss_function, tts)

            save_step = global_step + epoch_start
            if global_step % hp.save_step == 0:
                t.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       os.path.join(args.checkpoint_dir, 'checkpoint_ChunkGenNet_%d.pth.tar' % save_step))

            fig, axs = plt.subplots(3)
            fig.tight_layout(pad=0.5)
            fig.set_size_inches(18.5, 10.5, forward=True)
            axs[0].plot(np.array(train_losses), 'r')
            axs[1].plot(np.array(test_losses), 'r')
            axs[2].plot(np.array(gradients), 'r')
            plt.grid(True)

            axs[0].set_title('Training Loss')
            axs[1].set_title('Testing Loss')
            axs[2].set_title('Gradients')

            plt.grid(True)

            axs[0].set_xlabel('Epoch')
            axs[1].set_xlabel('Epoch')
            axs[2].set_xlabel('Epoch #batches')
            plt.grid(True)

            plt.savefig("loss_gradientsChunkNet.svg")
            matplotlib.pyplot.close()
            # -- Decay learning rate


def add_noise(
        waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor, lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r"""Scales and adds noise to waveform per signal-to-noise ratio.

    Specifically, for each pair of waveform vector :math:`x \in \mathbb{R}^L` and noise vector
    :math:`n \in \mathbb{R}^L`, the function computes output :math:`y` as

    .. math::
        y = x + a n \, \text{,}

    where

    .. math::
        a = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}}{10}} } \, \text{,}

    with :math:`\text{SNR}` being the desired signal-to-noise ratio between :math:`x` and :math:`n`, in dB.

    Note that this function broadcasts singleton leading dimensions in its inputs in a manner that is
    consistent with the above formulae and PyTorch's broadcasting semantics.

    .. devices:: CPU cpu

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
        noise (torch.Tensor): Noise, with shape `(..., L)` (same shape as ``waveform``).
        snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
        lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform`` and ``noise``, with shape
            `(...,)` (leading dimensions must match those of ``waveform``). If ``None``, all elements in ``waveform``
            and ``noise`` are treated as valid. (Default: ``None``)

    Returns:
        torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
        (same shape as ``waveform``).
    """

    if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim and (lengths is None or lengths.ndim == snr.ndim)):
        print(waveform.ndim)
        print(noise.ndim)
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

    # compute scale
    if lengths is not None:
        mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
            -1
        )  # (*, L) < (*, 1) = (*, L)
        masked_waveform = waveform * mask
        masked_noise = noise * mask
    else:
        masked_waveform = waveform
        masked_noise = noise

    energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
    energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise  # (*, L)


def plot_mel(mel, mel_pred, mel_target):
    """
    This function plots the three mel spectrograms of shape [1,features,frames]
    mel: the imput mel spectrogram with both synthetic and noisy speech joined together (has 2*num_frames size)
    mel_pred: The predicted mel spec from the model
    mel_target: The target mel
    """
    fig, axs = plt.subplots(4)
    axs[0].set_title('Mel_TTS')
    axs[0].set_ylabel('mel freq')
    axs[0].set_xlabel('frame')
    im = axs[0].imshow(librosa.power_to_db(mel[0, 0, :, :].cpu().squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[0])
    axs[1].set_title('Mel_Noisy')
    axs[1].set_ylabel('mel freq')
    axs[1].set_xlabel('frame')
    im = axs[1].imshow(librosa.power_to_db(mel[0, 1, :, :].cpu().squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[1])
    axs[2].set_title('Mel_pred')
    axs[2].set_ylabel('mel freq')
    axs[2].set_xlabel('frame')
    im = axs[2].imshow(librosa.power_to_db(mel_pred.detach()[0, :, :].cpu().squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[2])
    axs[3].set_title('Mel_clean')
    axs[3].set_ylabel('mel freq')
    axs[3].set_xlabel('frame')
    im = axs[3].imshow(librosa.power_to_db(mel_target.detach()[0, :, :].cpu().squeeze(0)), origin='lower',
                       aspect='auto')
    fig.colorbar(im, ax=axs[3])
    fig.tight_layout(pad=0.5)
    fig.set_size_inches(18.5, 10.5, forward=True)
    plt.savefig('melChunk.svg')
    matplotlib.pyplot.close()


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


def test(model, test_loader, mel_spectrogram, num_frames, loss_function, tts):
    """
    This cuntion computes the accumulated loss on the test set
    :param model: The model
    :param test_loader: the dataloader
    :param mel_spectrogram: the mel transformation
    :param num_frames: the # of frames of the target audio (including padding)
    :param loss_function: The loss function
    :param tts: The TTS
    :param chunk_size: The # of frames per chunk
    :return: the accumulates test loss
    """
    testing_loss = 0
    for i, data in enumerate(test_loader):
        model.eval()
        #file_path, embeds_path, text = data
        mel, mel_target = batch_mels_wav(data=data, mel_spectrogram=mel_spectrogram, num_frames=num_frames,
                                                    tts=tts)

        mel = mel.to(hp.device)
        mel_target = mel_target.to(hp.device)

        mel_pred = model(mel)  # forward pass

        # only consider the non-padded interval of the spec: padded columns are automatically mapped to zero
        mel_shift = mel[:, 1, :, :] + (-hp.pad_value)*torch.ones_like(mel[:, 1, :, :])
        non_empty_mask = mel_shift.abs().sum(dim=2).bool()
        mel_pred = torch.permute(mel_pred, (0, 2, 1))
        mel_pred[~non_empty_mask, :] = hp.pad_value
        mel_pred = torch.permute(mel_pred, (0, 2, 1))
        mel_target = mel_target.permute(0, 2, 1)
        mel = mel.permute(0, 1, 3, 2)
        # plot results
        plot_mel(mel, mel_pred, mel_target)
        # compute the loss
        test_loss = loss_function(mel_pred, mel_target, mel[:, 1, :, :], mel[:, 0, :, :])
        testing_loss = testing_loss + test_loss.item()
        model.train()
    return testing_loss


if __name__ == '__main__':
    main()
