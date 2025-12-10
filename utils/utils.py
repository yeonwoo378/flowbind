from multiprocessing.managers import Namespace
import os
import torch
import torchaudio

from typing import Dict, Optional, List, Tuple, Callable
from torchdiffeq import odeint
import numpy as np
import torch.nn as nn
from PIL import Image
import clip
from diffusers import DiffusionPipeline, AudioLDMPipeline
import torchvision.transforms as transforms
import yaml

from easydict import EasyDict as edict
import numpy as np
from model.models import *
from model.mlp_flows import MLPFlow
import json
from huggingface_hub import hf_hub_download

def ode_intergrate(ode_func: Callable,
                   init_x: torch.Tensor,
                   ode_opts: Dict = {},
                   method='euler10',
                   t_eps: float = 0,
                   init_t: float = 0.,
                   final_t: float = 1.,
                   t_arr: Optional[List[float]] = None,
                   intermediate_points: bool = False
                   ) -> torch.Tensor:

    class ODEFunc(torch.nn.Module):
        def __init__(self, ode_func):
            super(ODEFunc, self).__init__()
            self.ode_func = ode_func

        def forward(self, t, x):

            t = torch.ones_like(x[:, 0]) * t
            t = t.view(-1, 1)

            return self.ode_func(t, x)
    if t_arr is None and 'euler' == method:
        t = torch.FloatTensor([init_t-t_eps, final_t]).to(init_x.device)
    elif 'euler' in method:
        num_inference = int(method.split('euler')[-1])
        time_interval = (final_t - (init_t - t_eps)) / num_inference
        # t = torch.arange(0, num_inference) * time_interval + (init_t - t_eps)
        t = torch.arange(0, num_inference + 1) * \
            time_interval + (init_t - t_eps)
        t = t.to(init_x.device).float()
        method = 'euler'
    else:
        t = torch.FloatTensor([init_t-t_eps, final_t]).to(init_x.device)

    ode_function = ODEFunc(ode_func)

    z = odeint(ode_function, init_x, t, **
               {"atol": 1e-7, "rtol": 1e-9, "method": method, **ode_opts})  # dopri5

    if not intermediate_points:
        z = z[-1]
    return z


def denormalize(img):
    mean = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(
        1).unsqueeze(2).to(img.device)
    std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(
        1).unsqueeze(2).to(img.device)
    img = img * std + mean
    return img


def load_mean_std(modal, device):
    '''
    Load the precomputed stats for normalization/denormalization.
    '''
    assert modal in ['text', 'image',
                     'audio'], "modal should be image or text."

    if modal == 'text':
        text_mean = torch.load(
            f'samples/gemmae_text_mean.pt', map_location=device)
        text_std = torch.load(
            f'samples/gemmae_text_std.pt', map_location=device)
        return text_mean, text_std

    elif modal == 'image':
        img_mean = torch.load(
            'samples/unclip_img_mean.pt', map_location=device)
        img_std = torch.load('samples/unclip_img_std.pt', map_location=device)
        return img_mean, img_std

    elif modal == 'audio':
        audio_mean = torch.load(
            'samples/clap_audio_mean.pt', map_location=device)
        audio_std = torch.load(
            'samples/clap_audio_std.pt', map_location=device)
        return audio_mean, audio_std

    raise ValueError(f"modal {modal} is not supported.")


@torch.no_grad()
def load_text_enc_and_dec(name: str,
                          device: torch.device,
                          ):

    if name.startswith('gemmae'):
        # embedding gemma + gemma3 1b, get prefix len from name
        prefix_len = int(name.split('_')[-1]) if '_' in name else 3

        # 0. load ckpt and set inference mode
        device = 'cuda'
        dtype = torch.bfloat16
        ckpt_path = hf_hub_download(repo_id='yeonwoo378/flowbind', filename='gemmae_3.pt')
        ckpt = torch.load(ckpt_path, map_location='cpu')

        # 1. encoder model
        from sentence_transformers import SentenceTransformer  # 5.1.0
        enc_model = SentenceTransformer("google/embeddinggemma-300m",
                                        device=str(device),
                                        model_kwargs={"dtype": dtype})
        enc_model.eval()

        # 2. latent connector, first part of decoder
        class PrefixProjector(nn.Module):
            def __init__(self, in_dim: int, out_dim: int, prefix_len: int):
                super().__init__()
                hid = max(in_dim, out_dim)
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hid),
                    nn.Tanh(),
                    nn.Linear(hid, prefix_len * out_dim),
                )
                self.ln = nn.LayerNorm(out_dim)
                self.prefix_len = prefix_len
                self.out_dim = out_dim

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = self.net(x).view(x.size(0), self.prefix_len, self.out_dim)
                return self.ln(y)

        latent_projector = PrefixProjector(
            in_dim=768, out_dim=1152, prefix_len=prefix_len).to(device)
        latent_projector.load_state_dict(ckpt['prefix_proj'])
        latent_projector.eval()

        # 3. Gemma LM, remaining part of decoder
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dec_model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-1b-pt",
            dtype=dtype,
            device_map=device,
            attn_implementation='eager',
        )
        dec_model.load_state_dict(ckpt['model'])
        dec_model.eval()
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-pt")

        # 4. mean /std : normalizing / unnormalizing
        mean_pt, std_pt = load_mean_std('text', device)

        # 5. now wrap it all up
        @torch.autocast(device_type=device, dtype=dtype)
        def encode_feat(model, x, normalize=True):
            x = model.encode(x, convert_to_tensor=True,
                             normalize_embeddings=False)
            if normalize:
                x = (x - mean_pt) / std_pt
            return x.float().detach()

        @torch.autocast(device_type=device, dtype=dtype)
        def decode_feat(enc_vec, tokenizer, latent_projector, dec_model, strategy='greedy', denormalize=True):
            assert strategy == 'greedy', "Currently only greedy decoding is supported."
            if denormalize:
                enc_vec = enc_vec * std_pt + mean_pt

            prefix = latent_projector(enc_vec)
            bos_id = tokenizer.bos_token_id

            bos_ids = torch.full((enc_vec.shape[0], 1), bos_id, device=device)
            bos_emb = dec_model.get_input_embeddings()(bos_ids).to(dtype)
            inputs_embeds = torch.cat([prefix, bos_emb], dim=1)
            attn = torch.ones(
                (enc_vec.shape[0], prefix_len + 1), device=device, dtype=torch.long)

            gen_ids = getattr(dec_model, "module", dec_model).generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                max_new_tokens=64,
                do_sample=False,
                top_p=None,
                top_k=None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            decoded = [tokenizer.decode(
                ids, skip_special_tokens=True) for ids in gen_ids]
            return decoded

        return lambda x: encode_feat(enc_model, x, normalize=True), \
            lambda x: decode_feat(
                x, tokenizer, latent_projector, dec_model, denormalize=True)

    else:
        raise NotImplementedError(
            f"Representation {name} is not implemented.")


@torch.no_grad()
def load_image_enc(name: str,
                   device: torch.device,
                   model_type: str = 'mlp'
                   ):
    """
    Load the image encoder and preprocessing function based on the specified representation.

    Args:
        rep (str): The representation type (e.g., 'clipvitb32_image').
        device (torch.device): The device to load the model onto.
    """

    if name == "clipvitl14_image":
        model, preprocess = clip.load("ViT-L/14", device=device)
        model = model.to(device)
        encoder = model.encode_image
        mean_pt, std_pt = load_mean_std('image', device)
        return lambda x: (encoder(x.to(device)).float().detach() - mean_pt) / std_pt

    else:
        raise NotImplementedError(
            f"Image encoder {name} is not implemented.")


def load_image_dec(name, device):
    if name == 'stable-unclip-small':
        pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-unclip-small", torch_dtype=torch.float16, local_files_only=True
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        mean_pt, std_pt = load_mean_std(
            'image', device=device)

        def unclip_decode_image(pipe, images_embs):
            images_embs = images_embs.to(device).detach().float()
            images_embs = images_embs * std_pt + mean_pt
            bs = images_embs.shape[0]
            images = pipe(prompt=[""]*bs,
                          image_embeds=images_embs.half(),
                          output_type='pt').images
            return images

        return lambda x: unclip_decode_image(pipe, x.to(device))

    else:
        raise NotImplementedError(f"Image decoder {name} is not implemented.")


def load_audio_enc(name: str, device: torch.device):
    """
    Load the audio encoder based on the specified name.

    Args:
        name (str): The name of the audio encoder.
        device (torch.device): The device to load the model onto.

    Returns:
        Callable: A function that encodes audio inputs.
    """
    assert name == 'clap', "Currently only 'clap' audio encoder is supported."
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    model = model.to(device)
    model_ckpt = hf_hub_download(repo_id='yeonwoo378/flowbind', filename='clap.pt')
    model.load_ckpt(
        model_ckpt, verbose=False)

    model.eval()
    audio_ae = LightWeightAE(
        input_dim=512, hidden_dim=768, output_dim=768).to(device)
    audio_ae_path = hf_hub_download(repo_id='yeonwoo378/flowbind', filename='lightweight_audio_ae.pth')
    audio_ae.load_state_dict(torch.load(audio_ae_path, map_location=device))
    audio_ae.eval()
    del audio_ae.decoder
    audio_mean, audio_std = load_mean_std('audio', device)

    # return lambda x: model.get_text_embedding(x, use_tensor=True)
    def encode_audio(model, audio_ae, audio_data: torch.Tensor) -> torch.Tensor:
        """
        Encode audio data using the CLAP model.

        Args:
            model (CLAP_Module): The CLAP model.
            audio_data (torch.Tensor): The input audio data tensor.

        Returns:
            torch.Tensor: The encoded audio features.
        """
        audio_data = audio_data.to(device)
        # resample to 48000
        # audio_data = torchaudio.functional.resample(audio_data, orig_freq=16000, new_freq=48000)

        audio_features = model.get_audio_embedding_from_data(
            audio_data, use_tensor=True).float()
        audio_features = audio_ae.encoder(
            audio_features)  # (B, 512) -> (B, 768)

        audio_features = (audio_features - audio_mean) / audio_std  # Normalize

        return audio_features.detach()

    return lambda x: encode_audio(model, audio_ae, x.to(device))


def load_audio_dec(name: str, device: torch.device):
    """
    Load the audio encoder based on the specified name.

    Args:
        name (str): The name of the audio encoder. Default as audioldm-m-full.
        device (torch.device): The device to load the model onto.

    Returns:
        Callable: A function that encodes audio inputs.
    """
    if name == 'audioldm-m-full':
        repo_id = "cvssp/audioldm-m-full"
        model = AudioLDMPipeline.from_pretrained(
            repo_id, torch_dtype=torch.float16, local_files_only=True).to(device)
        model.set_progress_bar_config(disable=True)

        audio_ae = LightWeightAE(
            input_dim=512, hidden_dim=768, output_dim=768).to(device)
        audio_ae_path = hf_hub_download(repo_id='yeonwoo378/flowbind', filename='lightweight_audio_ae.pth')
        audio_ae.load_state_dict(torch.load(audio_ae_path, map_location=device))
        audio_ae.eval()
        del audio_ae.encoder

        audio_mean, audio_std = load_mean_std(
            'audio', device=device)

        def decode_audioldm(model, audio_ae, audio_embs: torch.Tensor) -> torch.Tensor:
            """
            Decode audio embeddings using the AudioLDM model.

            Args:
                model (AudioLDMPipeline): The AudioLDM model.
                audio_embs (torch.Tensor): (B, 512)

            Returns:
                torch.Tensor: The decoded audio waveform.
            """
            # denormalize
            audio_embs = (audio_embs * audio_std +
                          audio_mean).to(device)  # (B, 768)
            audio_embs = audio_ae.decoder(
                audio_embs).to(torch.float16)  # (B, 768)

            return model(prompt_embeds=audio_embs,
                         num_inference_steps=20,
                         ).audios

        return lambda x: decode_audioldm(model, audio_ae, x.to(device))

    else:
        raise NotImplementedError(f"Audio decoder {name} is not implemented.")


def get_cossim(
    x: torch.Tensor,
    y: torch.Tensor,
    as_batch: bool = True,
) -> torch.Tensor:
    """
    Compute the cosine similarity between two tensors.

    Args:
        x (torch.Tensor): First tensor.
        y (torch.Tensor): Second tensor.
        as_batch (bool): If True, aggregate cosine similarity over the batch dimension with mean.
                         If False, return cosine similarity for each sample in the batch.

    Returns:
        torch.Tensor: Cosine similarity between x and y.
    """
    assert x.shape == y.shape, "Input tensors must have the same shape."
    x = x / (x.norm(dim=-1, keepdim=True))
    y = y / (y.norm(dim=-1, keepdim=True))
    cos_sim = (x * y).sum(dim=-1)
    if as_batch:
        return cos_sim.mean()
    return cos_sim


def init_mlpflow(config, ckpt, text=False, image=False, audio=False, eval_mode=True, device='cuda', verbose=False):
    '''
    Args:
        config: dict, if json path then load automatically
        ckpt: dict, if .pt path then load automatically. If None, then init only (but should be explicitly specified)
        text, image, audio: whether to load or not.
        eval_mode: whether to call .eval() after loading
        device: 'cuda' or 'cpu'
    Returns:
        (text_model, image_model, audio_model)
        - none if not loaded.
    '''
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = json.load(f)
    if isinstance(ckpt, str):
        ckpt = torch.load(ckpt, map_location='cpu')

    # get from config
    hidden_dim = config['hidden_dim']
    num_layers = config['num_layers']
    prior = config.get('prior', 'explicit_residual')
    t_cond = config.get('t_cond', 'add')

    def init_single():
        net = MLPFlow(in_channels=768,
                      time_embed_dim=256,
                      model_channels=hidden_dim,
                      num_res_blocks=num_layers,
                      prior_method=prior,
                      t_cond=t_cond,).to(device)

        return net

    # init models
    text_flow, image_flow, audio_flow = None, None, None
    if text:
        text_flow = init_single()
        if ckpt is not None:
            text_flow.load_state_dict(ckpt['model_state_dict']['text'])
            if verbose:
                print("Loaded text flow ckpt.")
        if eval_mode:
            text_flow = text_flow.eval()

    if image:
        image_flow = init_single()
        if ckpt is not None:
            image_flow.load_state_dict(ckpt['model_state_dict']['image'])
            if verbose:
                print("Loaded image flow ckpt.")
        if eval_mode:
            image_flow = image_flow.eval()

    if audio:
        audio_flow = init_single()
        if ckpt is not None:
            audio_flow.load_state_dict(ckpt['model_state_dict']['audio'])
            if verbose:
                print("Loaded audio flow ckpt.")
        if eval_mode:
            audio_flow = audio_flow.eval()

    return text_flow, image_flow, audio_flow



def time_now(do_format=True):
    import datetime
    KST = datetime.timezone(datetime.timedelta(hours=9))

    now = datetime.datetime.now(KST)
    if do_format:
        return now.strftime('%Y-%m-%d_%H-%M-%S')
    return now


def count_parameters(model):
    flow_param_cnt = 0
    enc_param_cnt = 0
    for name, param in model.named_parameters():
        if 'logvar' in name:
            enc_param_cnt += param.numel()
        else:
            flow_param_cnt += param.numel()
    return flow_param_cnt, enc_param_cnt
