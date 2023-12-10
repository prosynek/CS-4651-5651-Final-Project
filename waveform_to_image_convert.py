import random
import pandas as pd
import torchaudio
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import random
from tqdm import tqdm
# from tqdm.auto import tqdm
tqdm.pandas()
DATA_DIR = './asvspoof_2sec_augment'
CLASSES = ['spoof', 'bonafide']
SAMPLE_RATE = 16000


def pad_trunc_audio(waveform, sample_rate, target_duration):
    n_rows, n_input_samples = waveform.shape
    # max number of samples (for 16 kHz * 3 s = 48000 samples)
    max_samples = sample_rate * target_duration

    # truncate audio to target length
    if (n_input_samples > max_samples):
        # takes first <target_duration> seconds of audio clip
        augmented_waveform = waveform[:, :max_samples]
    elif (n_input_samples < max_samples):
        # pad audio to target length
        # number of samples (zeros) needed to acheive <max_samples> number of samples
        n_zeros_pad = max_samples - n_input_samples
        pad_begin_len = random.randint(0, n_zeros_pad)
        pad_end_len = n_zeros_pad - pad_begin_len

        # pad with 0s
        pad_begin = torch.zeros((n_rows, pad_begin_len))
        pad_end = torch.zeros((n_rows, pad_end_len))
        augmented_waveform = torch.cat((pad_begin, waveform, pad_end), 1)
    return augmented_waveform


def transform_and_save(sample_path, sample_label, transform, transform_name, no_transform=False):
    parent_dir = f'./{transform_name}'

    # create directory to save images if doesn't already exist
    if not os.path.isdir(parent_dir):
        os.mkdir(parent_dir)
        os.mkdir(f'{parent_dir}/{CLASSES[0]}')
        os.mkdir(f'{parent_dir}/{CLASSES[1]}')

    a_id = sample_path.split('.flac')[-2].split('/')[-1]
    save_path = f'{parent_dir}/{sample_label}/{a_id}.png'
    
    if not os.path.exists(save_path):
        # load waveform from raw audio
        waveform, _ = torchaudio.load(sample_path)
        waveform = pad_trunc_audio(waveform, SAMPLE_RATE, 3)
       # print(librosa.get_duration(y=waveform, sr=SAMPLE_RATE))
    
        if not no_transform:
            # transform waveform to spectrogram
            spectrogram_tensor = transform(waveform)
            plt.figure(figsize=(10, 5))
            # show spectrogram plot - power converted to deccibels for better visualization
            plt.imshow(librosa.power_to_db(spectrogram_tensor)[
                       0, :, :], cmap='viridis', aspect='auto', origin='lower')
        else:
            plt.figure(figsize=(10, 5))
            num_channels, num_frames = waveform.shape
            time = np.arange(0, num_frames) / SAMPLE_RATE
            plt.plot(time, waveform.numpy()[0])
    
        # display(Audio(waveform.numpy()[0], rate=SAMPLE_RATE))
    
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    file_map_df = pd.read_csv(f'{DATA_DIR}/dataset_labels.csv')
    transform_name = 'waveform'
    transform = None
    file_map_df.progress_apply(lambda sample: transform_and_save(sample_path=f'{DATA_DIR}/{sample[0]}', sample_label=sample[1], transform_name=transform_name, transform=transform, no_transform=True), axis=1)
    print('waveforms complete')

