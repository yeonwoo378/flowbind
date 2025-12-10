import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp
# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py

import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat


def timestep_embedding(timesteps, dim, max_period=10, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)



def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def modulate(x, shift, scale):
    # return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    return x * (1 + scale) + shift

def get_prior_layers(version='v1', in_channels=768, model_channels=2048):
    if version == 'v1': # 17.83M 6-layer MLP variant
        layers = nn.Sequential(
            nn.Linear(in_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, 2*in_channels),
            nn.ReLU(),
            nn.Linear(2*in_channels, 2*in_channels),
            nn.ReLU(),
            nn.Linear(2*in_channels, 2*in_channels),
        )
    elif version == 'ln':
        # add layernorms, 17.85M
        layers = nn.Sequential(
            nn.Linear(in_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, 2*in_channels),
            nn.ReLU(),
            nn.LayerNorm(2*in_channels),
            nn.Linear(2*in_channels, 2*in_channels),
            nn.ReLU(),
            nn.LayerNorm(2*in_channels),
            nn.Linear(2*in_channels, 2*in_channels),
        )
    
    elif version == 'small':
        # four-layer MLP variant, 13.11M
        layers = nn.Sequential(
            nn.Linear(in_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, 2*in_channels),
        )
    elif version == 'large':
        # 29M 8-layer, add regularization as it is deeper
        layers = nn.Sequential(
            nn.Linear(in_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            nn.LayerNorm(model_channels),
            nn.Linear(model_channels, 2*in_channels),
        )
    elif version == 'residual':
        # 29.9M, 
        layers = nn.Sequential(
            nn.Linear(in_channels, model_channels),
            ResBlockNoTime(channels=model_channels, mid_channels=model_channels, emb_channels=0, dropout=0.1),
            ResBlockNoTime(channels=model_channels, mid_channels=model_channels, emb_channels=0, dropout=0.1),
            ResBlockNoTime(channels=model_channels, mid_channels=model_channels, emb_channels=0, dropout=0.1),
            nn.Linear(model_channels, 2*in_channels),
        )
    elif version == 'none':
        layers = nn.Identity()
    else:
        raise ValueError(f'Not recognized prior version {version}')

    return layers


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param mid_channels: the number of middle channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    """

    def __init__(
        self,
        channels,
        mid_channels,
        emb_channels,
        dropout,
        use_context=False,
        context_channels=512
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout

        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, mid_channels, bias=True),
        )
        if self.emb_channels > 0:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, mid_channels, bias=True),
            )
        else:
            self.emb_layers = None
            assert use_context == False, 'context is only available when emb_channels > 0'
        self.out_layers = nn.Sequential(
            nn.LayerNorm(mid_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Linear(mid_channels, channels, bias=True)
            ),
        )

        self.use_context = use_context
        if use_context:
            self.context_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(context_channels, mid_channels, bias=True),
        )

    def forward(self, x, emb, context):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb) if self.emb_layers is not None else 0.0
        if self.use_context:
            context_out = self.context_layers(context)
            h = h + emb_out + context_out
        else:
            if self.emb_layers is not None:
                emb_out = emb_out.squeeze(1).squeeze(1)
                h = h + emb_out
        h = self.out_layers(h)
        return x + h


class ResBlockNoTime(ResBlock):
    def __init__(self, channels, mid_channels, emb_channels, dropout, use_context=False, context_channels=512):
        assert emb_channels == 0, 'emb_channels must be 0 for ResBlockNoTime'
        assert not use_context, 'use_context must be False for ResBlockNoTime'
        super().__init__(channels, mid_channels, emb_channels, dropout, use_context, context_channels)
    
    def forward(self, x):
        return super().forward(x, emb=None, context=None)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class DiTMLPBlock(nn.Module):
    """
    A DiT-style block thtat uses adaLN-Zero but no attention.
    """
    def __init__(self, channels=2048, mid_channels=2048, emb_channels=256, dropout=0., mlp_ratio=1, emb_layer_type='linear', **block_kwargs):
        super().__init__()
        assert channels == mid_channels, 'for MLP block, channels must be equal to mid_channels'

        if emb_layer_type == 'linear':
             # if we have emb_channels != mid_channels (or if we just want to add a layer)
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_channels, mid_channels, bias=True),
            )
        elif emb_layer_type == 'identity':
             # if we already have emb_channels == mid_channels
            self.emb_layers = nn.Identity()

        self.norm = nn.LayerNorm(channels, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=mid_channels, hidden_features=int(mid_channels * mlp_ratio), drop=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            zero_module(nn.Linear(mid_channels, 3 * mid_channels, bias=True))
        )

    def forward(self, x, c, context=None):
        c = self.emb_layers(c.squeeze(1).squeeze(1)) # sinusoidal -> silu -> linear
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=1) # B x 3C
        x = x + gate_mlp * self.mlp(modulate(self.norm(x), shift_mlp, scale_mlp))
        return x



class MLPFlow(nn.Module):
    """
    DiT-style MLP blocks with adaLN-Zero.
    """

    def __init__(
        self,
        in_channels,
        time_embed_dim,
        model_channels,
        # bottleneck_channels, # intentionaly do this to fail fast
        # out_channels,
        num_res_blocks,
        dropout=0,
        use_context=False,
        context_channels=512,
        prior_method='explicit_residual',
        t_cond='add' # 'no', 'add', 'adaln'
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = in_channels # out_channels
        self.time_embed_dim = time_embed_dim
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.t_cond = t_cond

        assert self.t_cond == 'adaln'
        assert prior_method.startswith('explicit')

        self.time_embed = nn.Sequential(
            nn.Linear(time_embed_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(DiTMLPBlock(
                channels=model_channels, # 768
                mid_channels=model_channels, # also 768 
                emb_channels=time_embed_dim, # 256 but not used
                dropout=dropout, # 0 but not used
                mlp_ratio=4, # 768 -> 3072 -> 768 transformer-style mlp
                emb_layer_type='identity'
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.out = nn.Linear(model_channels, in_channels)
        
        if prior_method == 'explicit':
            version = 'v1'
        else:
            version = prior_method[len('explicit_'):] # e.g. explicit_v2 -> v2
        self.logvar = get_prior_layers(version=version, in_channels=in_channels, model_channels=model_channels)
        

    def forward(self, timesteps, x, context=None, y=None, return_var=False, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        x = x.squeeze()
        x = self.input_proj(x) # identity
        t_emb = timestep_embedding(timesteps, self.time_embed_dim, repeat_only=False) # 256-dim sinusoidal
        emb = self.time_embed(t_emb) # 256 -> 2048-dim -> 2048dim

        for block in self.res_blocks:
            x = block(x, emb.unsqueeze(1), context)

        return self.out(x)


class SimpleAEMLP(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        bottleneck_channels,
        # out_channels,
        num_res_blocks,
        dropout=0,
        use_context=False,
        context_channels=512,
        prior_method='onestep',
    ):
        super().__init__()

        self.image_size = 1
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = in_channels # out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout


        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                bottleneck_channels,
                0, #time_embed_dim,
                dropout,
                use_context=use_context,
                context_channels=context_channels
            ))

        self.res_blocks = nn.ModuleList(res_blocks)

        self.out = nn.Sequential(
            nn.LayerNorm(model_channels, eps=1e-8),
            nn.SiLU(),
            zero_module(nn.Linear(model_channels, self.out_channels, bias=True)),
        )
        
        self.fc1 = nn.Linear(in_channels, in_channels)

        self.logvar = nn.Sequential(
            nn.Linear(in_channels, model_channels),
            nn.ReLU(),
            nn.Linear(model_channels, model_channels),
            nn.ReLU(),
            zero_module(nn.Linear(model_channels, in_channels))
        )
        if prior_method == 'explicit':
            self.logvar = nn.Sequential(
                nn.Linear(in_channels, model_channels),
                nn.ReLU(),
                nn.Linear(model_channels, model_channels),
                nn.ReLU(),
                nn.Linear(model_channels, model_channels),
                nn.ReLU(),
                nn.Linear(model_channels, 2*in_channels),
                nn.ReLU(),
                nn.Linear(2*in_channels, 2*in_channels),
                nn.ReLU(),
                nn.Linear(2*in_channels, 2*in_channels),
            )
        


    def forward(self, x, context=None, y=None, return_var=False, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        x = x.squeeze()
        # context = context.squeeze()
        x = self.input_proj(x)
    
        for block in self.res_blocks:
            x = block(x, None, context)


        return self.out(x)
        
  
