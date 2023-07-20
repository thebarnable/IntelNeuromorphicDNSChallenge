import numpy as np
import torch
import torchaudio.transforms
from datetime import datetime
from utility.snr import si_snr

def collate_fn(batch):
    noisy, clean, noise = [], [], []

    for sample in batch:
        noisy += [torch.FloatTensor(sample[0])]
        clean += [torch.FloatTensor(sample[1])]
        noise += [torch.FloatTensor(sample[2])]

    return torch.stack(noisy), torch.stack(clean), torch.stack(noise)


def stft_splitter(audio, window, n_fft=512):
    with torch.no_grad():
        audio_stft = torch.stft(audio,
                                n_fft=n_fft,
                                onesided=True,
                                return_complex=True, window=window)
        return audio_stft.abs(), audio_stft.angle()


def stft_mixer(stft_abs, stft_angle, window, n_fft=512):
    return torch.istft(torch.complex(stft_abs * torch.cos(stft_angle),
                                     stft_abs * torch.sin(stft_angle)),
                       n_fft=n_fft, onesided=True, window=window)


def calculate_filter_banks(n_fft, n_filter_banks, sample_rate, device):
    with torch.no_grad():
        f_l, f_u = 0, sample_rate/2
        def m_fn(f): return 1125*np.log(1+f/700)
        def m_inv_fn(m): return 700*(np.exp(m/1125) - 1)
        m_vals = [(m_fn(f_u) - m_fn(f_l))/(n_filter_banks+1) * i + m_fn(f_l)
                  for i in range(n_filter_banks+2)]
        f_vals = [np.floor((m_inv_fn(m) * (n_fft + 1))/sample_rate)
                  for m in m_vals]
        H_vals = np.zeros((n_filter_banks, n_fft//2+1), dtype=np.float32)

        map = {k: [] for k in range(n_fft//2+1)}
        for i in range(1, n_filter_banks+1):
            for k in range(n_fft//2+1):
                if f_vals[i-1] < k <= f_vals[i]:
                    H_vals[i-1, k] = (k-f_vals[i-1]) / \
                        (f_vals[i]-f_vals[i-1])
                    map[k].append(i-1)
                if f_vals[i] < k < f_vals[i+1]:
                    H_vals[i-1, k] = (f_vals[i+1]-k) / \
                        (f_vals[i+1]-f_vals[i])
                    map[k].append(i-1)

        return torch.from_numpy(H_vals).to(device), map


def reconstruct_wave_from_mfcc(n_fft, abs, phase, y_vals, y_vals_network, H_vals, map, inv_spec_transformation):
    x_vals = torch.clone(abs)

    for m in range(1, n_fft // 2):
        val = 0
        for i in range(len(map[m])):
            val += (y_vals_network[:, map[m][i], :] - y_vals[:, map[m][i], :] +
                    H_vals[map[m][i], m] * abs[:, m, :]) / H_vals[map[m][i], m]

        x_vals[:, m, :] = val/len(map[m])
    x_vals[x_vals < 0] = 0

    return inv_spec_transformation(torch.complex(x_vals * torch.cos(phase),
                                                 x_vals * torch.sin(phase))), x_vals


def prob(p):
    rand = np.random.random()
    return rand < p


def vol_scaling(data_noisy, data_clean, sr):
    val = np.random.randint(-20, 0)
    transform = torchaudio.transforms.Vol(val, 'db')
    return transform(data_noisy), transform(data_clean)


def lowpass(data_noisy, data_clean, sr):
    cutoff = np.random.randint(3500, 4500)
    return torchaudio.functional.lowpass_biquad(data_noisy, sr, cutoff), torchaudio.functional.lowpass_biquad(data_clean, sr, cutoff)


def resample(data_noisy, data_clean, sr):
    new_freq = np.random.randint(int(0.75*sr/1000), int(1.25*sr/1000))
    if new_freq == sr//1000:
        return data_noisy, data_clean
    return torchaudio.functional.resample(data_noisy, sr//1000, new_freq), torchaudio.functional.resample(data_clean, sr//1000, new_freq)


def time_stretch(data_noisy, data_clean, sr):
    val = np.random.uniform(0.5, 1.5)
    transform = torchaudio.transforms.TimeStretch()
    spec = torchaudio.transforms.Spectrogram(power=None)
    inv_spec = torchaudio.transforms.InverseSpectrogram()
    return inv_spec(transform(spec(data_noisy), val)), inv_spec(transform(spec(data_clean), val))


def clipping(data_noisy, data_clean, sr):
    val = np.random.uniform(0.8, 1) * torch.max(data_noisy)
    data_noisy[data_noisy > val] = val
    data_clean[data_clean > val] = val
    return data_noisy, data_clean


def nop_stats(dataloader, stats, sub_stats, print=True):
    t_st = datetime.now()
    for i, (noisy, clean, noise) in enumerate(dataloader):
        with torch.no_grad():
            noisy = noisy
            clean = clean

            score = si_snr(noisy, clean)
            sub_stats.correct_samples += torch.sum(score).item()
            sub_stats.num_samples += noisy.shape[0]

            processed = i * dataloader.batch_size
            total = len(dataloader.dataset)
            time_elapsed = (datetime.now() - t_st).total_seconds()
            samples_sec = time_elapsed / (i + 1) / dataloader.batch_size
            header_list = [f'Train: [{processed}/{total} '
                           f'({100.0 * processed / total:.0f}%)]']
            if print:
                stats.print(0, i, samples_sec, header=header_list)