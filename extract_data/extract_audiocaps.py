import os
from data import AudioCapsDataset
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text_enc_name = 'gemmae'  

dataset = AudioCapsDataset()
print(f"Loaded dataset with {len(dataset)} samples.")
SAVE_DIR = f'./feats/audiocaps'  # Update this path accordingly

audio_enc = load_audio_enc('clap', device=device)
text_enc, _ = load_text_enc_and_dec(text_enc_name, device=device)
os.makedirs(SAVE_DIR, exist_ok=True)

start_idx = 0
end_idx = len(dataset)
for idx in tqdm(range(start_idx, end_idx)):
    data = dataset[idx]
    text = data[0]
    audio = data[2]
    fname = os.path.join(SAVE_DIR, f'{idx}.npz')
    if os.path.exists(fname):
        print(f"Skip existing {idx}")
        continue

    if text is None:
        print(f"Skipping sample {idx} due to missing text.")
        print(f"text: {text} \n audio: {audio}")
        continue

    with torch.no_grad():
        audio = audio.mean(dim=0, keepdim=True).to(device)

        audio_feat = audio_enc(audio)
        text_feat = text_enc([text])

    audio_feat = audio_feat.cpu().squeeze(0).numpy()
    text_feat = text_feat.cpu().squeeze(0).numpy()

    # save features
    np.savez_compressed(fname, audio=audio_feat, text=text_feat)
