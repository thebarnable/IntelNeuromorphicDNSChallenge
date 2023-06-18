import torch.nn as nn
import torch
import torch.nn.functional as F
import hyperparams as hp


class ChunkNet(nn.Module):
    """
    This model predicts the mask applied to the surrounding chunks of each chunk.
    """
    def __init__(self, input_len, num_layers, hidden_size, output_len, num_chunks_per_process):
        """
            :param input_len: The number of input neurons
            :param num_layers: The number of fully connected layers
            :param hidden_size: The number of hidden neurons per layer
            :param output_len: The number of output neurons
            :param num_chunks_per_process: The number of chunks which are processed per Channel (per audio).
            F.e. if I process the chunk to be predicted and the surrounding ones, the number is 3
            """
        super(ChunkNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_len = input_len
        self.output_len = output_len
        self.num_chunks_per_process = num_chunks_per_process

        self.norm = nn.BatchNorm1d(hidden_size)
        #self.linear_in = nn.Linear(768, hidden_size)
        self.linear_in = nn.Linear(2*64 * hp.chunk_size * hp.num_chunks_per_process, hidden_size) #uncomment this and comment above to remove convolution

        self.linear_h = nn.Linear(hidden_size, hidden_size)
        self.act = nn.ReLU()
        # The linear layer that maps from hidden state space to mel
        self.linear_out = nn.Linear(hidden_size, output_len)

        # convolution layers
        self.conv1 = nn.Conv2d(in_channels=2 * self.num_chunks_per_process, out_channels=16, kernel_size=(3, 3),
                               stride=(1, 1))
        self.bn1 = nn.GroupNorm(1, 16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(1, 1))
        self.bn2 = nn.GroupNorm(1, 32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1))
        self.bn3 = nn.GroupNorm(1, 64)
        self.act2 = nn.ELU()
        self.pooling22 = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.act3 = nn.LeakyReLU()
        self.act4 = nn.Sigmoid()
        self.norm_out = nn.BatchNorm1d(output_len)

    def forward(self, mel):  # mel is of shape [B,C,T,F]
        mel_split = mel.split(hp.chunk_size, dim=2)
        mel_out = []
        # split the mel in chunks
        for idx, frame in enumerate(mel_split):  # N, 1
            # concat for each chunk the surrounding chunks for each audio
            if (idx < int(self.num_chunks_per_process/2)):
                list = []
                for i in range(self.num_chunks_per_process):
                    list.append(mel_split[i])
                frame = torch.cat((list), dim=1)
            elif (idx >= (len(mel_split) - 1) - int(self.num_chunks_per_process/2)):
                list = []
                for i in range(self.num_chunks_per_process):
                    list.append(mel_split[idx - i])
                list.reverse()
                frame = torch.cat(list, dim=1)
            else:
                list = []
                for i in range(self.num_chunks_per_process):
                    list.append(mel_split[idx + i + 1 - int(self.num_chunks_per_process / 2)])
                frame = torch.cat(list, dim=1)  # [B,2*C,num_chunks,T]

            #apply convolution
            x1 = self.pooling22(self.act2(self.bn1(self.conv1(frame))))
            x2 = self.pooling22(self.act2(self.bn2(self.conv2(x1))))
            x3 = self.act2(self.bn3(self.conv3(x2)))
            #frame_flat = x3.view(-1, 768)
            frame_flat = frame.view(-1, 2*64*hp.chunk_size*hp.num_chunks_per_process) #uncomment this and comment above to remove convolution
            x = self.linear_in(frame_flat)
            x = self.norm(x)
            x = self.act3(x)
            for i in range(self.num_layers):
                x = self.linear_h(x)
                x = self.norm(x)
                x = self.act3(x)
                x = self.dropout(x)
            out = (torch.abs(self.linear_out(x)))

            norm_factor = torch.sum(out, dim=1).unsqueeze(1)
            out = out / norm_factor
            # out has shape Bx[a_chunk_Ch1_1, a_chunk_Ch2_1, a_chunk_Ch1_2, a_chunk_Ch2_2, ...]

            # apply the mask to every sample in the batch
            mel_apnd = []
            for i in range(frame.size(0)):
                #this is just a matrix-vector multiplication
                product = 0
                for j in range(out.size(1)):
                    product = product + (frame[i, j, :, :] * out[i, j])
                mel_apnd.append(product.unsqueeze(0))
            mel_apnd = torch.cat(mel_apnd, dim=0)
            mel_out.append(mel_apnd)
        mel_out = torch.cat(mel_out, dim=1)
        mel_out = mel_out.permute(0, 2, 1)  # [B,F,T]
        return mel_out


if __name__ == '__main__':
    # for testing
    a = torch.rand(2, 2, 800, 64)
    m = ChunkNet(4 * hp.chunk_size * 2 * hp.num_mels, hp.layers_DNN, hp.hidden_size_DNN, 2 * 4, 4)
    b = m(a)
    print("This model has " + str(sum(p.numel() for p in m.parameters() if p.requires_grad)) + " parameters")

