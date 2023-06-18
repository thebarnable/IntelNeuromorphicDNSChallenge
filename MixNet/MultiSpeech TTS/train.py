from data import get_batch_loader
from data_loaders import TextLoader
from loss import Loss
from model import Model
from optim import AdamWarmup
from padder import get_padders
from pipelines import get_pipelines
from tokenizer import CharTokenizer
from torch.utils.data import DataLoader
from interfaces import ITrainer
from torch.nn import Module
from pathlib import Path
from typing import Union
from torch import Tensor
import numpy as np

import torchaudio.transforms as T
import torchaudio
import os
import os.path as osp

import torch.nn.functional as F
import librosa
from dnsmos import DNSMOS



from args import (
    get_args,
    get_model_args,
    get_loss_args,
    get_optim_args,
    get_aud_args,
    get_data_args,
    get_trainer_args
)
import os
import torch


class Trainer(ITrainer):
    def __init__(
            self,
            train_loader: DataLoader,
            test_loader: DataLoader,
            model: Module,
            criterion: Module,
            optimizer: object,
            save_dir: Union[str, Path],
            steps_per_ckpt: int,
            epochs: int,
            last_step: int,
            device: str
            ) -> None:
        super().__init__()
        self.save_dir = save_dir
        self.epochs = epochs
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = model
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.steps_per_ckpt = steps_per_ckpt
        self.last_step = last_step

    def set_train_mode(self):
        self.model = self.model.train()

    def set_test_mode(self):
        self.model = self.model.test()

    def _predict(self, text: Tensor, spk: Tensor, speech: Tensor):
        text = text.to(self.device)
        spk = spk.to(self.device)
        speech = speech.to(self.device)
        return self.model(
            text, spk, speech
        )

    def _train_step(
            self,
            speech: Tensor,
            speech_length: Tensor,
            mask: Tensor,
            text: Tensor,
            spk: Tensor
            ):
        mel_results, stop_results, alignments = self._predict(
            text, spk, speech
        )
        self.optimizer.zero_grad()
        loss = self.criterion(
            lengths=speech_length,
            mask=mask,
            stop_pred=stop_results,
            mels_pred=mel_results,
            mels_target=speech,
            alignments=alignments
        )
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self):
        total_train_loss = 0
        for item in self.train_loader:
            loss = self._train_step(*item)
            total_train_loss += loss
        return total_train_loss / len(self.train_loader)

    def test(self):
        total_test_loss = 0
        for (speech, speech_length, mask, text, spk,_) in self.test_loader:
            mel_results, stop_results, alignments = self._predict(
                text, spk, speech
            )
            #print(mel_results.size())
            total_test_loss += self.criterion(
                lengths=speech_length,
                mask=mask,
                stop_pred=stop_results,
                mels_pred=mel_results,
                mels_target=speech,
                alignments=alignments
                ).item()
        return total_test_loss / len(self.test_loader)

    def fit(self):
        # TODO: Add per step exporting here
        # TODO: Add tensor board here
        self.model.train()
        try:
            for epoch in range(self.epochs):
                train_loss = self.train()
                test_loss = self.test()
                test_loss=0
                print(
                    'epoch={}, training loss: {}, testing loss: {}'.format(
                        epoch, train_loss, test_loss)
                    )
                self.save_ckpt(epoch)   
        except Exception as e: 
            print('Error: '+ str(e))
            self.save_ckpt(101)   
            
            
    def predictText(self, args: dict):
        count = 0
        
        for (speech, speech_length, mask, text, spk,file_path) in self.test_loader:
            count = count + 1
            mel_results, stop_results, alignments = self._predict(
                text, spk, speech
            )

            text = text.to(self.device)
            spk = spk.to(self.device)
            speech = speech.to(self.device)

            mel_results, stop_results, alignments = self.model(text, spk, speech)
            
            mel_results=torch.permute(mel_results, (0, 2, 1))
            mel_results=mel_results.to(self.device)
            print("mel_results: "+str(mel_results.size()))
            
            n_stft = int(((args.n_fft)//2) + 1)
            inverse_transform = T.InverseMelScale(sample_rate=args.sampling_rate, n_stft=n_stft, n_mels=args.n_mels).to(self.device)
            grifflim_transform = T.GriffinLim(n_fft=args.n_fft).to(self.device)

            inverse_waveform = inverse_transform(mel_results.detach())
            inverse_waveform=inverse_waveform.to(self.device)
            audio = grifflim_transform(inverse_waveform).to('cpu')
            
            torchaudio.save(filepath="./samples/prediction_audio" + str(count) + ".wav", src=audio,
                            sample_rate=args.sampling_rate)
            #print("filepath=./samples/prediction_audio" + str(count) + ".wav")

        
   

    def save_ckpt(self, idx: int):
        path = os.path.join(self.save_dir, f'ckptBigSet_{idx}')
        torch.save(self.model, path)
        print(f'checkpoint saved to {path}')


def get_model(args: dict, model_args: dict):
    return Model(
        **model_args
    )


def get_optim(args: dict, opt_args: dict, model: Module):
    return AdamWarmup(parameters=model.parameters(), **opt_args)


def get_criterion(args: dict, criterion_args: dict):
    return Loss(**criterion_args)


def get_tokenizer(args):
    # TODO: refactor this code
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


def get_trainer(args: dict):
    # TODO: refactor this code
    tokenizer = get_tokenizer(args)
    vocab_size = tokenizer.vocab_size
    data = TextLoader(args.train_path).load().split('\n')
    n_speakers = len(set(map(lambda x: x.split(args.sep)[0], data)))
    device = args.device
    model_args = get_model_args(
        args,
        vocab_size,
        tokenizer.special_tokens.pad_id,
        n_speakers
        )
    loss_args = get_loss_args(args)
    optim_args = get_optim_args(args)
    aud_args = get_aud_args(args)
    data_args = get_data_args(args)
    trainer_args = get_trainer_args(args)
    model = get_model(args, model_args).to(device)
    optim = get_optim(args, optim_args, model)
    criterion = get_criterion(args, loss_args)
    text_padder, aud_padder = get_padders(0, tokenizer.special_tokens.pad_id)
    audio_pipeline, text_pipeline = get_pipelines(tokenizer, aud_args)
    train_loader = get_batch_loader(
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
    
    return Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optim,
        last_step=0,
        **trainer_args
    )
    
def predict_from_model(model: Module):
    count = 0      
    for (speech, speech_length, mask, text, spk,file_path) in trained_net.test_loader:
        count = count + 1

        file_path=" ".join([x for x in file_path])
        text = text.to('cuda')
        spk = spk.to('cuda')
        speech = speech.to('cuda')

        mel_results, stop_results, alignments = net(text, spk, speech)
        
        mel_results=torch.permute(mel_results, (0, 2, 1))
        mel_results=mel_results.to('cuda')
        print("mel_results: "+str(mel_results.size()))
        
        n_stft = int(((args.n_fft)//2) + 1)
        inverse_transform = T.InverseMelScale(sample_rate=args.sampling_rate, n_stft=n_stft, n_mels=args.n_mels).to('cuda')
        grifflim_transform = T.GriffinLim(n_fft=args.n_fft).to('cuda')

        inverse_waveform = inverse_transform(mel_results.detach())
        inverse_waveform=inverse_waveform.to('cuda')
        audio = grifflim_transform(inverse_waveform).to('cpu')
        noisy=audio.to('cuda')
        noisy=torch.permute(noisy,(1,0))
        waveform, in_sr = librosa.load(file_path)
        noisy=F.pad(audio, (waveform.shape[0] - noisy.size(0), 0))
        
        # SNR
        SNR=si_snr(waveform,noisy)
        print(SNR)
        # dnsmos
        dnsmos = DNSMOS()
        quality = dnsmos(audio)  # It is in order [ovrl, sig, bak]
        print(quality)
        
        torchaudio.save(filepath="./samples/prediction_audio" + str(count) + ".wav", src=audio,
                        sample_rate=args.sampling_rate)
                        
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

if __name__ == '__main__':
    args = get_args()
    #torch.backends.cudnn.enabled = False #solves error: https://github.com/pytorch/captum/issues/564
    
    
    #----------TRAINING-----------
    trainer=get_trainer(args)
    #trainer.fit()
    print(sum(p.numel() for p in trainer.model.parameters()))
    print(sum(p.numel() for p in trainer.model.parameters() if p.requires_grad ))
    
    
    #----------TRAINING OF EXISTING-----------
    #trainer=get_trainer(args)
    #folder = "models"
    #save_network = osp.join("./", folder)
    #filename_load = "ckpt_9"
    #net=(torch.load(osp.join(save_network, filename_load)))
    #trainer.model=net
    #trainer.fit()


    #folder = "models"
    #save_network = osp.join("./", folder)
    #filename_save = "trainerBigSet_tb.pth.tar"
    #torch.save(trainer, osp.join(save_network, filename_save))



    #----------SYTHESIS-----------
    #folder = "models"
    #save_network = osp.join("./", folder)
    #filename_load = "ckpt_9"
    #trained_net=get_trainer(args)
    #trained_net=(torch.load(osp.join(save_network, filename_load)))
    #net=(torch.load(osp.join(save_network, filename_load))).to('cuda')
    #trained_net.model=net
    #trained_net.predictText(args)
    #predict_from_model(net)
 

