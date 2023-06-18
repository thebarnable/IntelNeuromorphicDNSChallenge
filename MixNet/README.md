# MixNet

This is a PyTorch implementation of [Deep speech denoising by ASR-TTS resynthesis](link.pdf). The corresponding paper is located in the repository. The dataloader and MultiSpeech TTS model was taken from [msalhab96/MultiSpeech](https://github.com/msalhab96/MultiSpeech) but hte TTS model must be trained further. I added the function to synthesize own .wav files with a pretrained model for MultiSpeech TTS. However, the pretrained TTS used for training MixNet is the TTS from [coqui-ai/TTS](https://github.com/coqui-ai/tts). I used GriffinLim as Vocoder. The DNSMOS and SI-SNR functions were taken from the [Intel Neuromorphic DNS Challenge](https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge/tree/main).


# Train on your data
In order to train the model on your data, follow the steps below 
# 1. data preprocessing 
* prepare your data and make sure the data is formatted in an PSV format as below without the header
```
speaker reference audio path,audio path,text,duration
path/speaker_reference.wav|path/clean_audio_sample.wav|the text in that file|3.2 
```
# 2. Setup development environment
* create enviroment 
```bash
python -m venv env
```
* activate the enviroment
```bash
source env/bin/activate
```
* install the required dependencies
```bash
pip install -r requirements.txt
```
# 3. Training 
* args does pass the data loader arguments to the dataloader (batch size and the paths where the training_data.txt files are stored)
* hyperparams stores the arguments for training the model and the model itself (number of layers, learning rate, number of neurons, chunk size, number of chunks, etc.).
* update the args (for the dataloader) and hyperparams (for the model) file if needed
* train the model 

# 4. Evaluating
* update the args (for the dataloader): set batch size to 1
* run eval<model_name>.py
