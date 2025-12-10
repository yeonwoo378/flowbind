import os
import time
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision import transforms


class LaionCOCOAesCachedMMDataset(Dataset):
    def __init__(self, root, force_float32=True, cache='all'):  # cache: 'all' or 'none'
        self.root = Path(root)
        self.files = sorted([p for p in self.root.glob("*.npz")])
        self.n = len(self.files)

        self.image_np = None
        self.text_np = None

        if cache == 'all':
            imgs, txts = [], []
            for p in self.files:
                with np.load(p, allow_pickle=False) as z:
                    img = z['image']
                    txt = z['text']
                    if force_float32:
                        if img.dtype != np.float32:
                            img = img.astype(np.float32, copy=False)
                        if txt.dtype != np.float32:
                            txt = txt.astype(np.float32, copy=False)
                    imgs.append(img)
                    txts.append(txt)

            # Prefer stack when shapes are uniform; else keep lists (avoid object arrays).
            try:
                self.image_np = np.stack(imgs, axis=0)  # fast contiguous
                self.text_np = np.stack(txts, axis=0)
            except ValueError:
                self.image_np = imgs
                self.text_np = txts

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.image_np is not None:
            # cached path
            img = self.image_np[idx]
            txt = self.text_np[idx]
            return txt, img, None
        else:
            # lazy path (better for RAM / many workers)
            p = self.files[idx]
            with np.load(p, allow_pickle=False) as z:
                img = z['image'].astype(np.float32, copy=False)
                txt = z['text'].astype(np.float32, copy=False)
            return txt, img, None


class T2ICachedMMDataset(LaionCOCOAesCachedMMDataset):
    def __init__(self, root, force_float32=True, cache='all'):
        super().__init__(root, force_float32, cache)


class LaionCOCOAesMMDataset(Dataset):
    def __init__(self, cache_dir, metadata_file, transform=None):

        image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        self.cache_dir = cache_dir
        self.transform = image_transform if transform is None else transform

        # Load the metadata and create a mapping from hash to caption
        metadata_path = os.path.join(metadata_file)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Please run the metadata creation script first."
            )

        self.metadata = pd.read_csv(metadata_path)

        self.image_paths = [
            os.path.join(self.cache_dir, f"{h}.jpg") for h in self.metadata['sha256']
        ]
        self.image_paths = np.array(self.image_paths)
        self.metadata.set_index('sha256', inplace=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Get the image path
        img_path = self.image_paths[idx]
        img_sha256 = os.path.splitext(os.path.basename(img_path))[0]

        caption = self.metadata.loc[img_sha256, 'caption']

        if not isinstance(caption, str):
            del img_path, img_sha256, caption

            return self.__getitem__((idx + 1) % len(self))

        try:
            image = Image.open(img_path).convert("RGB")

        except FileNotFoundError:
            print(f"Warning: Image file not found at {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        return caption, image, None
