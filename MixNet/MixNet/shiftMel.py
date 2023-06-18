import hyperparams as hp
import torch


def shiftMel(mel_in,shift,chunk_size):
    mel_in = mel_in.permute(0, 1, 3, 2)
    mel_split = mel_in.split(chunk_size, dim=2)
    mel_out = []
    num_chunks_per_process = 1
    for idx, frame in enumerate(mel_split):  # N, 1

        if (idx < int(num_chunks_per_process / 2)):  # if ==0
            list = []
            for i in range(num_chunks_per_process):
                list.append(mel_split[i])
            frame = torch.cat((list), dim=1)
        elif (idx <= (len(mel_split) - shift)):
            list = []
            for i in range(num_chunks_per_process):
                list.append(mel_split[idx + 1 - int(hp.num_chunks_per_process / 2)])
            frame = torch.cat(list, dim=1)  # [B,2*C,num_chunks,T]

        out = (torch.rand(mel_in.size(0), num_chunks_per_process))

        norm_factor = torch.sum(out, dim=1).unsqueeze(1)
        out = out / norm_factor
        # out has shape Bx[a_chunk_Ch1_1, a_chunk_Ch2_1, a_chunk_Ch1_2, a_chunk_Ch2_2, ...]

        if (idx <= (len(mel_split) - shift)):
            mel_apnd = []
            for i in range(frame.size(0)):
                product = 0
                for j in range(out.size(1)):
                    product = product + (frame[i, j, :, :] * out[i, j])
                mel_apnd.append(product.unsqueeze(0))
            mel_apnd = torch.cat(mel_apnd, dim=0)
            mel_out.append(mel_apnd)

    pad = torch.zeros_like(mel_out[0])

    mel_out = torch.cat(mel_out, dim=1)
    for i in range(shift-1):
        mel_out = torch.cat((pad, mel_out), dim=1)

    mel_out = mel_out.permute(0, 2, 1)
    return mel_out
