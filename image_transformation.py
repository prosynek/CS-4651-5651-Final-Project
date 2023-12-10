# https://github.com/BakingBrains/Sound_Classification/blob/main/dataset.py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import random
import librosa
import pandas as pd
import torchaudio
import os

class ASVSpoofDataset(Dataset):
    # audio_df : dataframe of format {abs_path_to_audio_sample : sample_label, }
    def __init__(self, audio_df, sample_rate, transform):
        self.data = audio_df  
        self.transform = transform
        self.sample_rate = sample_rate

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        # index = index of audio sample (row index)
        row = self.data.iloc[index, :]
        sample_path = row.iloc[0]   # get the path to the audio sample
        sample_label = row.iloc[1]
        sample_label = 1 if sample_label == 'bonafide' else 0
        waveform, _ = torchaudio.load(sample_path)                          # read audio file
        waveform, _ = librosa.effects.trim(waveform, top_db=30)             # trim beginning & ending silences
        waveform = self._pad_trunc_audio(waveform, self.sample_rate, 3)     # pad/truncate clip to be consistent length
        transformed_waveform = self.transform(waveform)                     # apply transformation (spectrogram)
        return transformed_waveform, sample_label

    def _pad_trunc_audio(self, waveform, sample_rate, target_duration):
        n_rows, n_input_samples = waveform.shape
        max_samples = sample_rate * target_duration                 # max number of samples (for 16 kHz * 3 s = 48000 samples)

        # truncate audio to target length
        if (n_input_samples > max_samples):
            augmented_waveform = waveform[:, :max_samples]          # takes first <target_duration> seconds of audio clip
        elif (n_input_samples < max_samples):
            # pad audio to target length
            n_zeros_pad = max_samples - n_input_samples             # number of samples (zeros) needed to acheive <max_samples> number of samples
            pad_begin_len = random.randint(0, n_zeros_pad)
            pad_end_len = n_zeros_pad - pad_begin_len
            
            # pad with 0s
            pad_begin = torch.zeros((n_rows, pad_begin_len))
            pad_end = torch.zeros((n_rows, pad_end_len))
            augmented_waveform = torch.cat((pad_begin, waveform, pad_end), 1)
        return augmented_waveform
    
    
    

if __name__ == '__main__':
    SAMPLE_RATE = 16000
    spectrogram_transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=128, win_length=1024) 
    # spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=128, win_length=1024)
    # spectrogram_transform = torchaudio.transforms.MFCC(sample_rate=SAMPLE_RATE, n_mfcc=256, melkwargs={"n_fft": 2048, "hop_length": 256, 'n_mels' : 256})
    
    data_df = pd.read_csv('./dataset/full_dataset.csv', index_col=0)
    print(f'length of dataset df = {len(data_df)}')
    dataset = ASVSpoofDataset(data_df, SAMPLE_RATE, spectrogram_transform)
    print(f'length of dataset = {len(dataset)}')

    print('saving spectrograms...')
    for i, data in enumerate(dataset):
        sample, label = data[0], data[1]
        # sample_1, label_1 = training_dataset[10]
        # print(f'sample spectrogram = {sample_1}\nshape = {sample_1.shape}\nsample label = {label_1}')
        plt.figure(figsize=(5, 4))
        # plt.imshow(sample.log2()[0,:,:].numpy(), cmap='viridis') # Mel Spectrogram
        plt.imshow(librosa.power_to_db(sample)[0,:,:], cmap='viridis', aspect='auto', origin='lower')
        # plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
        if label == 1:
            save_path = './asvspoof_mel_spectrograms/bonafide/bf_'
        else: 
            save_path = './asvspoof_mel_spectrograms/spoof/sp_'

        plt.savefig(f'{save_path}{i}.png', bbox_inches='tight')
        plt.close()
    