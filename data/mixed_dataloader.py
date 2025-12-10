import os

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .audiocaps import  AudioCapsCachedMMDataset
from .laion_coco_aes import LaionCOCOAesCachedMMDataset,T2ICachedMMDataset
from .vggsound import VGGSoundCachedMMDataset


def cache_mixed_collate_fn(batch, dtype=torch.float32):
    # batch: list of (text, image, audio), each may be None
    texts, images, audios = zip(*batch)
    n = len(batch)

    # 1) Convert once per sample (or keep None)
    t_list = [None if x is None else torch.as_tensor(x, dtype=dtype) for x in texts]
    i_list = [None if x is None else torch.as_tensor(x, dtype=dtype) for x in images]
    a_list = [None if x is None else torch.as_tensor(x, dtype=dtype) for x in audios]

    # 2) Build index lists once
    ti_idx = [k for k in range(n) if t_list[k] is not None and i_list[k] is not None]
    ia_idx = [k for k in range(n) if i_list[k] is not None and a_list[k] is not None]
    ta_idx = [k for k in range(n) if t_list[k] is not None and a_list[k] is not None]

    ret = {'ti': {}, 'ta': {}, 'ia': {}, 'tia': {}}

    if ti_idx:
        ret['ti']['text']  = torch.stack([t_list[k] for k in ti_idx], 0)
        ret['ti']['image'] = torch.stack([i_list[k] for k in ti_idx], 0)
    if ia_idx:
        ret['ia']['image'] = torch.stack([i_list[k] for k in ia_idx], 0)
        ret['ia']['audio'] = torch.stack([a_list[k] for k in ia_idx], 0)
    if ta_idx:
        ret['ta']['text']  = torch.stack([t_list[k] for k in ta_idx], 0)
        ret['ta']['audio'] = torch.stack([a_list[k] for k in ta_idx], 0)

    return ret

class MixedDataset(Dataset):
    """
    Custom PyTorch Dataset that combines multiple datasets.
    This dataset can handle audio, image, and text data.
    """

    def __init__(self, datasets, audio_transform=None, image_transform=None):
        """
        Args:
            datasets (list): List of datasets to combine.
            audio_transform (callable, optional): Transform to be applied on an audio sample.
            image_transform (callable, optional): Transform to be applied on an image sample.
            text_transform (callable, optional): Transform to be applied on a text sample.
        """
        self.datasets = datasets
        self.audio_transform = audio_transform
        self.image_transform = image_transform

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for dataset in self.datasets:
            if idx < len(dataset):
                 return dataset[idx]
            idx -= len(dataset)
        raise IndexError("Index out of range")

    
def load_cached_mm_dataset(dataset_list, cache_paths=None):
    mixed_dataset = []
    if cache_paths is None:
        cache_paths = [''] * len(dataset_list)
    assert len(dataset_list) == len(cache_paths), "dataset_list and cache_paths must have the same length"
    for dataset_name, cache_path in zip(dataset_list, cache_paths):
        if dataset_name == 'audiocaps':
            cache_path = 'feats/audiocaps' if cache_path == '' else cache_path
            mixed_dataset.append( AudioCapsCachedMMDataset(root=cache_path))
        elif dataset_name == 'vggsound':
            cache_path = 'feats/vggsound' if cache_path == '' else cache_path
            mixed_dataset.append(VGGSoundCachedMMDataset(root=cache_path))
        elif dataset_name == 'laion':
            cache_path = 'feats/laion' if cache_path == '' else cache_path
            mixed_dataset.append(LaionCOCOAesCachedMMDataset(root=cache_path))
        elif dataset_name == 'flickr30k':
            cache_path = 'feats/flickr30k' if cache_path == '' else cache_path
            mixed_dataset.append(T2ICachedMMDataset(root=cache_path))
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        print(f"Loaded {dataset_name} with {len(mixed_dataset[-1])} samples from {cache_path}")

    mixed_dataset = MixedDataset(mixed_dataset)
    return mixed_dataset