import os
from torch.utils.data import Dataset

import numpy as np

from pathlib import Path


class VGGSoundCachedMMDataset(Dataset):
    """
    Custom PyTorch Dataset for the AudioCaps dataset with multimodal support.
    This version caches processed audio and image data to speed up training.
    """

    def __init__(self, root):

        if root.endswith('-all.npz'):
            # load already-concatenated npz file
            cached_file = np.load(root, allow_pickle=False)
            self.image_np = cached_file['image_np']
            self.audio_np = cached_file['audio_np']
            self.root = self.files = None
            assert len(self.image_np) == len(self.audio_np)
            self.n = len(self.image_np)

            return

        self.root = Path(root)

        self._image_np = []
        self._audio_np = []

        for file in os.listdir(self.root):
            if file.endswith(".npz"):
                with np.load(os.path.join(self.root, file)) as data:
                    self._image_np.append(data['image'])
                    self._audio_np.append(data['audio'])
        self.audio_np = np.array(self._audio_np)
        self.image_np = np.array(self._image_np)

        del self._audio_np, self._image_np

    def __len__(self):
        return 183728

    def __getitem__(self, idx):

        return None, self.image_np[idx], self.audio_np[idx]
