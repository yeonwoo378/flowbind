import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os
from torchvision import transforms
import numpy as np
from pathlib import Path
import csv
from PIL import Image


class AudioCapsCachedMMDataset(Dataset):
    """
    Custom PyTorch Dataset for the AudioCaps dataset with multimodal support.
    This version caches processed audio and image data to speed up training.
    """

    def __init__(self, root):
        
        self.root = Path(root)
        # sort and aggreagate audio and text
        self._audio_np = []
        self._text_np = []

        for file in os.listdir(self.root):
            with np.load(os.path.join(self.root, file)) as data:
                if 'audio' in data:
                    self._audio_np.append(data['audio'])
                if 'text' in data:
                    self._text_np.append(data['text'])

        self.audio_np = np.array(self._audio_np)
        self.text_np = np.array(self._text_np)
        del self._audio_np, self._text_np

    def __len__(self):
        return len(self.audio_np)

    def __getitem__(self, idx):
        return self.text_np[idx], None, self.audio_np[idx]


class AudioCapsDataset(Dataset):

    def __init__(self, csv_file='/path/to/audiocaps/audio/train.csv', 
                 audio_path='/path/to/audiocaps/audio', 
                 target_sample_rate=48000,
                 return_yid=False, only_first_caption=True):
        self.csv_file = csv_file
        self.audio_path = audio_path
        self.target_sample_rate = target_sample_rate
        self.return_yid = return_yid
        self.only_first_caption = only_first_caption

        self.dataset = self.load_dataset()
        # audiocap_id,youtube_id,start_time,caption

    def load_dataset(self):
        dataset = []
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_file = os.path.join(self.audio_path, row['audio_filename'])
                dataset.append({
                    'yt_id' : row['youtube_id'],
                    'audio': audio_file,
                    'caption': row['caption'],
                    'st': row['start_time'] 
                })

        if self.only_first_caption:
            print("[!] Using only_first_caption=True for loading AudioCaps dataset, loading only the first among five captions as an valid sample.")
            appeared = set()
            filtered_dataset = []
            for item in dataset:
                if item['yt_id'] not in appeared:
                    appeared.add(item['yt_id'])
                    filtered_dataset.append(item)
            dataset = filtered_dataset

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        file_path = f"{sample['yt_id']}_{sample['st']}.wav"

        audio_file_path = os.path.join(self.audio_path, file_path)
        if not os.path.exists(audio_file_path) :
            print(f"Audio file not found: {audio_file_path}")
            return None, None, None

        audio_file, sr = torchaudio.load(audio_file_path)
        
        # resample to 48000
        audio_file = torchaudio.transforms.Resample(sr, self.target_sample_rate)(audio_file)

        caption = sample['caption']
        if self.return_yid:
            return caption, None, audio_file, sample['yt_id']

        return caption, None, audio_file