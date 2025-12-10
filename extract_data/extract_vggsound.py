import os
from data import VGGSoundDataset
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from utils.utils import *
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

VGG_TRAIN_DIR = '/path/to/vggsound/train'  # Update this path accordingly


def main(args):
    # dataloader for batch processing
    DATASET = args.dataset
    SAVE_DIR = f'./feats/{DATASET}'

    print(f"Will save to {SAVE_DIR}")

    custom_collate_fn = None

    if DATASET == 'vggsound':
        dataset = VGGSoundDataset(VGG_TRAIN_DIR,
                                  transform=None,
                                  target_sr=48000)
    else:
        raise NotImplementedError

    # _, clip_preprocess = clip.load('ViT-L/14', device='cpu')

    print("loading encoders...")
    image_enc = load_image_enc(IMAGE_ENC, device=device)
    audio_enc = load_audio_enc('clap', device=device)
    os.makedirs(SAVE_DIR, exist_ok=True)

    raw_indices = list(range(len(dataset)))
    selected_indices = raw_indices

    # import ipdb; ipdb.set_trace()
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn=custom_collate_fn,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=4)

    i = 0
    for batch in tqdm(dataloader):
        image = batch[0]
        audio = batch[1]

        with torch.no_grad():
            assert image.ndim == 4  # B, C, H, W
            image_feat = image_enc(image)
            audio_feat = audio_enc(audio)

        image_feat = image_feat.cpu().numpy()
        audio_feat = audio_feat.cpu().numpy()

        for _image, _audio in zip(image_feat, audio_feat):
            idx = selected_indices[i]
            i += 1
            save_name = os.path.join(SAVE_DIR, f'{idx}.npz')
            if os.path.exists(save_name):
                print(f"Skip existing {idx}")
                continue
            np.savez_compressed(save_name, image=_image, audio=_audio)

    assert i == len(
        selected_indices), f"Should be equal: {i} vs {len(selected_indices)}"
    print(f"Saved {i} samples to {SAVE_DIR}")

    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--dataset', type=str, default='vggsound',
                        choices=['vggsound'], help='Dataset to use')
    args = parser.parse_args()
    main(args)
