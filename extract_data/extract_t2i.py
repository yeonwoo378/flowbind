import os
from data import LaionCOCOAesMMDataset
import numpy as np
import torch
from tqdm import tqdm
import clip
from utils.utils import *
from torch.utils.data import DataLoader
from datasets import load_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LAION_COCO_CSV = '/path/to/laion-coco035-metadata.csv'  # Update this path accordingly
LAION_COCO_AESTHETIC_IMAGES = '/path/to/laion-coco-aesthetic-images'  # Update this path accordingly

def main(args):
    DATASET = args.dataset
    SAVE_DIR = f'./feats/{DATASET}'

    print(f"Will save to {SAVE_DIR}")

    def custom_collate_fn(batch): return {
        'text': [item[0] for item in batch],
        'image': [item[1] for item in batch],
        'audio': [item[2] for item in batch],
    }

    if DATASET == 'laion':
        dataset = LaionCOCOAesMMDataset(
            cache_dir=LAION_COCO_AESTHETIC_IMAGES,
            metadata_file=LAION_COCO_CSV,
            transform=None)

    elif DATASET == 'flickr30k':
        flickr_ds = load_dataset("nlphuji/flickr30k")
        dataset = flickr_ds['test']

        # dataloader for batch processing as the format differs
        def custom_collate_fn(batch): return {
            'text': [item['caption'][0] for item in batch],
            'image': [item['image'] for item in batch],
        }

    else:
        raise ValueError(f"Unknown dataset: {DATASET}")

    import clip
    clip_preprocess = clip.clip._transform(224)

    print("loading encoders...")
    text_enc, _ = load_text_enc_and_dec(TEXT_ENC, device=device)
    image_enc = load_image_enc(IMAGE_ENC, device=device)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # subsample for parallel processing
    raw_indices = selected_indices = list(range(len(dataset)))
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            collate_fn=custom_collate_fn,
                            shuffle=False,
                            pin_memory=True,
                            drop_last=False,
                            num_workers=NUM_WORKERS)

    i = 0
    for batch in tqdm(dataloader):
        text = batch['text']  # list of str
        image = batch['image']  # list of PIL images
        # audio = batch['audio'] is not used

        with torch.no_grad():
            image = torch.stack([clip_preprocess(_img)
                                for _img in image]).to(device)
            assert image.ndim == 4  # B, C, H, W
            text_feat = text_enc(text)
            image_feat = image_enc(image)

        text_feat = text_feat.cpu().numpy()
        image_feat = image_feat.cpu().numpy()

        for _text, _image in zip(text_feat, image_feat):
            idx = selected_indices[i]
            i += 1
            save_name = os.path.join(SAVE_DIR, f'{idx}.npz')
            if os.path.exists(save_name):
                print(f"Skip existing {idx}")
                continue
            np.savez_compressed(save_name, text=_text, image=_image)

    assert i == len(
        selected_indices), f"Should be equal: {i} vs {len(selected_indices)}"
    print(f"Saved {i} samples to {SAVE_DIR}")

    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='flickr30k', choices=[
                        'flickr30k', 'laion'], help='Dataset to use')

    args = parser.parse_args()
    main(args)
