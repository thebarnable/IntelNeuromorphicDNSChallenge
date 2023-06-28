import librosa
import librosa.display
import matplotlib.pyplot as plt


# List of WAV files
#audio_files = ["debug_train_0_clean.wav", "debug_train_0_noisy.wav", "debug_train_0_output.wav"]
#audio_files = ["debug_train_1_clean.wav", "debug_train_1_noisy.wav", "debug_train_1_output.wav"]
#audio_files = ["debug_train_2_clean.wav", "debug_train_2_noisy.wav", "debug_train_2_output.wav"]
#audio_files = ["debug_train_res_clean.wav", "debug_train_res_noisy.wav", "debug_train_res_output.wav"]

#audio_files = ["debug_coeffs_train_0_clean.wav", "debug_coeffs_train_0_noisy.wav", "debug_coeffs_train_0_output.wav"]
#audio_files = ["debug_coeffs_train_1_clean.wav", "debug_coeffs_train_1_noisy.wav", "debug_coeffs_train_1_output.wav"]
#audio_files = ["debug_coeffs_train_2_clean.wav", "debug_coeffs_train_2_noisy.wav", "debug_coeffs_train_2_output.wav"]
#audio_files = ["debug_coeffs_train_res_clean.wav", "debug_coeffs_train_res_noisy.wav", "debug_coeffs_train_res_output.wav"]

epoch = 8
#audio_files = ["debug_coeffs_train_0_clean_{}.wav".format(epoch), "debug_coeffs_train_0_noisy_{}.wav".format(epoch), "debug_coeffs_train_0_output_{}.wav".format(epoch)]
#audio_files = ["debug_coeffs_train_1_clean_{}.wav".format(epoch), "debug_coeffs_train_1_noisy_{}.wav".format(epoch), "debug_coeffs_train_1_output_{}.wav".format(epoch)]
#audio_files = ["debug_coeffs_train_2_clean_{}.wav".format(epoch), "debug_coeffs_train_2_noisy_{}.wav".format(epoch), "debug_coeffs_train_2_output_{}.wav".format(epoch)]
#audio_files = ["debug_coeffs_train_res_clean_{}.wav".format(epoch), "debug_coeffs_train_res_noisy_{}.wav".format(epoch), "debug_coeffs_train_res_output_{}.wav".format(epoch)]

epoch = 6
audio_files = ["debug2_coeffs_train_0_clean_{}.wav".format(epoch), "debug2_coeffs_train_0_noisy_{}.wav".format(epoch), "debug2_coeffs_train_0_output_{}.wav".format(epoch)]
audio_files = ["debug2_coeffs_train_1_clean_{}.wav".format(epoch), "debug2_coeffs_train_1_noisy_{}.wav".format(epoch), "debug2_coeffs_train_1_output_{}.wav".format(epoch)]
audio_files = ["debug2_coeffs_train_2_clean_{}.wav".format(epoch), "debug2_coeffs_train_2_noisy_{}.wav".format(epoch), "debug2_coeffs_train_2_output_{}.wav".format(epoch)]
#audio_files = ["debug2_coeffs_train_res_clean_{}.wav".format(epoch), "debug2_coeffs_train_res_noisy_{}.wav".format(epoch), "debug2_coeffs_train_res_output_{}.wav".format(epoch)]

# Configure subplots
num_files = len(audio_files)
fig, axs = plt.subplots(num_files, 1, figsize=(10, 6*num_files))

# Iterate over audio files and plot spectrograms
for i, audio_path in enumerate(audio_files):
    # Load the WAV file
    waveform, sample_rate = librosa.load(audio_path)

    # Compute the STFT spectrogram
    stft = librosa.stft(waveform, n_fft=512, hop_length=160, win_length=400)

    # Convert the magnitude spectrogram to decibels
    spectrogram = librosa.amplitude_to_db(abs(stft))

    # Display the spectrogram
    axs[i].imshow(spectrogram, aspect="auto", origin="lower")
    axs[i].set_title(audio_path)
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Frequency")

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()
