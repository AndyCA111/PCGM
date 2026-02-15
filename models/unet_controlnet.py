import sys
sys.path.append(".")
import math
import copy
import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial
from opensora.models.ae.videobase import CausalVAEModel
from torch.utils import data
from pathlib import Path
from torch.optim import AdamW, Adam
from torchvision import transforms as T, utils
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from accelerate import DistributedDataParallelKwargs
import glob, os
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from dataloader import get_data_loaders
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import nibabel as nib
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import xformers, xformers.ops
from typing import Any, Dict, List, Optional, Tuple, Union
from safetensors.torch import load_file
# import sys

# a 3d medical control net 
# 



def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def decode_from_latent(latents, model):
    latents = (latents /0.188).to(model.dtype)
    recon_img = model.decode(latents)
    return recon_img
    
    
def save_nii(latents, model, save_path):
    latents = (latents /0.188).to(model.dtype)
    recon_img = model.decode(latents)
    recon_img = recon_img.cpu().detach().numpy()
    recon_img = recon_img[0].mean(axis=0)
    ni_img = nib.Nifti1Image(recon_img, affine=np.eye(4))
    print("sample save!", ni_img.shape)
    nib.save(ni_img, save_path)

def get_alpha_cum(t):
    return torch.where(t >= 0, torch.cos((t + 0.008) / 1.008 * math.pi / 2)**2, 1.0)

def get_z_t(x_0, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_cum)*x_0 + torch.sqrt(1-alpha_cum)*eps
    return x_t, eps


def get_eps_x_t(x_0, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    eps = (x_t - torch.sqrt(alpha_cum)*x_0)/torch.sqrt(1-alpha_cum)
    return eps

def get_w(t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    w = torch.sqrt(alpha_cum)/torch.sqrt(1-alpha_cum)
    return w

def get_v_x_t(x_0, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    v = (x_t - torch.sqrt(alpha_cum)*x_0)
    return v

def get_x0_x_t(eps, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    x_0 = (x_t - eps * torch.sqrt(1-alpha_cum)) / torch.sqrt(alpha_cum)
    return x_0

def get_x0_v(v, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    sigma = 1 - alpha_cum
    return x_t - sigma*v

def get_v_x0(x0, x_t, t):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    sigma = 1 - alpha_cum
    v = (x_t - x0)/sigma
    return v

def get_z_t_(x_0, t):
    alpha_cum = get_alpha_cum(t)[:,None]
    return torch.sqrt(alpha_cum)*x_0, torch.sqrt(1-alpha_cum)

def get_z_t_via_z_tp1(x_0, z_tp1, t, t_p1):
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    alpha_cum_p1 = get_alpha_cum(t_p1)[:, None, None, None, None]
    beta_p1 = 1 - alpha_cum_p1/alpha_cum
    mean_0 = torch.sqrt(alpha_cum)*beta_p1/(1-alpha_cum_p1)
    mean_tp1 = torch.sqrt(1-beta_p1)*(1-alpha_cum)/(1-alpha_cum_p1)

    var = (1-alpha_cum)/(1-alpha_cum_p1)*beta_p1

    return mean_0*x_0 + mean_tp1*z_tp1, var

def ddim_sample(x_0, z_tp1, t, t_p1):
    epsilon = get_eps_x_t(x_0, z_tp1, t_p1)
    alpha_cum = get_alpha_cum(t)[:, None, None, None, None]
    x_t = torch.sqrt(alpha_cum)*x_0 + torch.sqrt(1-alpha_cum)*epsilon
    return x_t

def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev

def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


# helpers functions

def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def create_custom_forward(module, return_dict=None):
    def custom_forward(*inputs):
        if return_dict is not None:
            return module(*inputs, return_dict=return_dict)
        else:
            return module(*inputs)

    return custom_forward

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])



class ControlNetConditioningEmbedding(nn.Module):
    """
    condition encoder make mask into same shape of latent.
    as input will be 160 192 160, output will be (40, 24, 20)
    """

    def __init__(
        self,
        conditioning_embedding_channels: int = 256,
        conditioning_channels: int = 1,
        block_out_channels: Tuple[int, ...] = (16, 32, 96, 256),
    ):
        super().__init__()

        self.conv_in = nn.Conv3d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            if i == len(block_out_channels) - 2:
                channel_in = block_out_channels[i]
                channel_out = block_out_channels[i + 1]
                self.blocks.append(nn.Conv3d(channel_in, channel_in, kernel_size=3, padding=1))
                self.blocks.append(nn.Conv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=(1, 2, 2)))
            else:
                channel_in = block_out_channels[i]
                channel_out = block_out_channels[i + 1]
                self.blocks.append(nn.Conv3d(channel_in, channel_in, kernel_size=3, padding=1))
                self.blocks.append(nn.Conv3d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = zero_module(
            nn.Conv3d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding
     


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (4, 4, 4), (2, 2, 2), (1, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class Block3d(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (3, 3, 3), padding=(1, 1, 1))
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock3d(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block3d(dim, dim_out, groups=groups)
        self.block2 = Block3d(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h c) x y -> (b h) (x y) c', h=self.heads)

        query = q.contiguous()
        key = k.contiguous()
        value = v.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)

        out = rearrange(hidden_states, '(b h) (x y) c -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)

class CrossAttention(nn.Module):
    def __init__(self, dim, dim_con, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Linear(dim_con, hidden_dim*2, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, kv=None):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')

        self.to_kv(kv)
        kv = torch.cat([kv.unsqueeze(dim=1)]*f, dim=1)
        kv = rearrange(kv, 'b f h c -> (b f) h c')
        k, v = self.to_kv(kv).chunk(2, dim=-1)
        k = rearrange(k, 'b d (h c) -> (b h) d c', h=self.heads)
        v = rearrange(v, 'b d (h c) -> (b h) d c', h=self.heads)

        q = self.to_q(x)
        q = rearrange(q, 'b (h c) x y -> (b h) (x y) c', h=self.heads)

        query = q.contiguous()
        key = k.contiguous()
        value = v.contiguous()
        hidden_states = xformers.ops.memory_efficient_attention(query, key, value, attn_bias=None)

        out = rearrange(hidden_states, '(b h) (x y) c -> b (h c) x y', h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b=b)


# attention along space and time

class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x,
            pos_bias=None,
            focus_present_mask=None
    ):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)


# model

class Unet3D(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            attn_heads=8,
            attn_dim_head=32,
            total_slices=256,
            use_bert_text_cond=False,
            init_dim=None,
            init_kernel_size=7,
            use_sparse_linear_attn=True,
            block_type='resnet',
            resnet_groups=8
    ):
        super().__init__()

        self.channels = channels
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(channels, init_dim, (init_kernel_size, init_kernel_size, init_kernel_size),
                                   padding=(init_padding, init_padding, init_padding))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        self.has_cond = exists(cond_dim)
        # self.has_cond = False
        self.null_cond_emb = nn.Parameter(torch.randn(1,500, cond_dim)) if self.has_cond else None

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        block_klass3d = partial(ResnetBlock3d, groups=resnet_groups)
        block_klass_cond3d = partial(block_klass3d, time_emb_dim=time_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                Downsample(dim_out) if not is_last else nn.Identity(),
                block_klass_cond(dim_out, dim_out),
                block_klass_cond3d(dim_out, dim_out),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn1 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn1 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn1 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn2 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn2 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn2 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn3 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn3 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn3 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn4 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn4 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn4 = block_klass_cond3d(mid_dim, mid_dim)

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                block_klass_cond(dim_out * 2, dim_in),
                Upsample(dim_in) if not is_last else nn.Identity(),
                block_klass_cond(dim_in, dim_in),
                block_klass_cond3d(dim_in, dim_in),
            ]))

        self.final_conv_0 = block_klass_cond3d(dim * 2, dim)
        self.final_conv_1 = nn.Sequential(
            nn.GELU(),
            nn.Conv3d(dim, channels, 1)
        )

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=2.,
            **kwargs
    ):
        logits = self.forward(*args, null_cond_prob=0., **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1., **kwargs)
        return [logits, null_logits]

    def forward(
            self,
            x,
            time,
            indexes=None,
            cond=None,
            mid_additional_res = None,
            down_additional_res_list = None,
            condition_scaling = 1.0,
            null_cond_prob=0.1,
            focus_present_mask=None,
            prob_focus_present=0.
            # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        is_controlnet = mid_additional_res is not None and down_additional_res_list is not None
        x = self.init_conv(x)

        r = x*1.0
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            null_cond_emb = torch.cat([self.null_cond_emb]*batch, dim=0)
            cond = torch.where(rearrange(mask, 'b -> b 1 1'), null_cond_emb, cond)

        h = []

        for idx,(block1, downsample, block2, temporal_block) in enumerate(self.downs):
            x = block1(x, t)
            h.append(x)
            x = downsample(x)
            x = block2(x, t)
            x = temporal_block(x, t)

        x = self.mid_block1(x, t)
        ###
        x = self.mid_spatial_attn1(x)
        x = self.mid_cross_attn1(x, kv=cond)
        x = self.mid_temporal_attn1(x, t)
        ###
        x = self.mid_spatial_attn2(x)
        x = self.mid_cross_attn2(x, kv=cond)
        x = self.mid_temporal_attn2(x, t)
        ###
        x = self.mid_spatial_attn3(x)
        x = self.mid_cross_attn3(x, kv=cond)
        x = self.mid_temporal_attn3(x, t)
        ###
        x = self.mid_spatial_attn4(x)
        x = self.mid_cross_attn4(x, kv=cond)
        x = self.mid_temporal_attn4(x, t)
        ###
        x = self.mid_block2(x, t)

        # add_controlnet
        if is_controlnet:
            x = x + mid_additional_res

        
        for block1, upsample, block2, temporal_block in self.ups:
            x = torch.cat((x, h.pop() + condition_scaling * down_additional_res_list.pop()), dim=1)
            x = block1(x, t)
            x = upsample(x)
            x = block2(x, t)
            x = temporal_block(x, t)

        x = torch.cat((x, r), dim=1)
        x = self.final_conv_0(x, t)
        x = self.final_conv_1(x)
        return x

class Controlnet3D(nn.Module):
    def __init__(
            self,
            dim,
            cond_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            attn_heads=8,
            attn_dim_head=32,
            total_slices=256,
            use_bert_text_cond=False,
            init_dim=None,
            init_kernel_size=7,
            use_sparse_linear_attn=True,
            block_type='resnet',
            resnet_groups=8
    ):
        super().__init__()

        self.channels = channels
        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.control_proj = ControlNetConditioningEmbedding(conditioning_channels = 1)
        self.init_conv = nn.Conv3d(channels, init_dim, (init_kernel_size, init_kernel_size, init_kernel_size),
                                   padding=(init_padding, init_padding, init_padding))
        # dimensions
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # text conditioning

        # self.has_cond = exists(cond_dim) or use_bert_text_cond
        self.has_cond = False
        self.null_cond_emb = nn.Parameter(torch.randn(1, 500, cond_dim)) if self.has_cond else None

        # layers

        self.downs = nn.ModuleList([])
        # control net zero conv layers
        self.controlnet_downs = nn.ModuleList([])

        num_resolutions = len(in_out)

        # block type

        block_klass = partial(ResnetBlock, groups=resnet_groups)
        block_klass_cond = partial(block_klass, time_emb_dim=time_dim)

        block_klass3d = partial(ResnetBlock3d, groups=resnet_groups)
        block_klass_cond3d = partial(block_klass3d, time_emb_dim=time_dim)

        # modules for all layers

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass_cond(dim_in, dim_out),
                
                Downsample(dim_out) if not is_last else nn.Identity(),
                
                block_klass_cond(dim_out, dim_out),
                block_klass_cond3d(dim_out, dim_out),
                
            ]))
            
            self.controlnet_downs.append(   
                    zero_module(nn.Conv3d(dim_out, dim_out, kernel_size=1))     
                )
            #control net zero conv module.
            
            # for _ in range(3):
            #     self.controlnet_downs.append(   
            #         zero_module(nn.Conv3d(dim_out, dim_out, kernel_size=1))     
            #     )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass_cond(mid_dim, mid_dim)

        spatial_attn = EinopsToAndFrom('b c f h w', 'b f (h w) c', Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn1 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn1 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn1 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn2 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn2 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn2 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn3 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn3 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn3 = block_klass_cond3d(mid_dim, mid_dim)
        ###
        self.mid_spatial_attn4 = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_cross_attn4 = Residual(PreNorm(mid_dim, CrossAttention(mid_dim, heads=attn_heads, dim_con=cond_dim)))
        self.mid_temporal_attn4 = block_klass_cond3d(mid_dim, mid_dim)

        self.mid_block2 = block_klass_cond(mid_dim, mid_dim)
        
        self.mid_control = zero_module(nn.Conv3d(mid_dim, mid_dim, kernel_size=1))     

        # for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
        #     is_last = ind >= (num_resolutions - 1)

        #     self.ups.append(nn.ModuleList([
        #         block_klass_cond(dim_out * 2, dim_in),
        #         Upsample(dim_in) if not is_last else nn.Identity(),
        #         block_klass_cond(dim_in, dim_in),
        #         block_klass_cond3d(dim_in, dim_in),
        #     ]))

        # self.final_conv_0 = block_klass_cond3d(dim * 2, dim)
        # self.final_conv_1 = nn.Sequential(
        #     nn.GELU(),
        #     nn.Conv3d(dim, channels, 1)
        # )
    def from_unet(self, unet3d):
        
        unet_state_dict = unet3d.state_dict()
    
        controlnet_state_dict = self.state_dict()

        new_state_dict = {}
        for name, param in controlnet_state_dict.items():
            if name in unet_state_dict:
                new_state_dict[name] = unet_state_dict[name]
            else:
                new_state_dict[name] = param

        self.load_state_dict(new_state_dict)
        
        return self

    def forward(
            self,
            x,
            time,
            indexes=None,
            cond=None,
            img_cond = None,
            null_cond_prob=0.1,
            focus_present_mask=None,
            prob_focus_present=0.
            # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'
        # preprocess
        x = self.init_conv(x)
        control_cond = self.control_proj(img_cond)
        # print(x.shape)
        # print(control_cond.shape)
        x = x + control_cond
        
        r = x * 1.0
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        # classifier free guidance
        if self.has_cond:
            batch, device = x.shape[0], x.device
            mask = prob_mask_like((batch,), null_cond_prob, device=device)
            null_cond_emb = torch.cat([self.null_cond_emb]*batch, dim=0)
            cond = torch.where(rearrange(mask, 'b -> b 1 1'), null_cond_emb, cond)
            
            
        down_res = []
        for idx, (block1, downsample, block2, temporal_block) in enumerate(self.downs):
            x = block1(x, t)
            down_res.append(x)
            x = downsample(x)
            x = block2(x, t)
            x = temporal_block(x, t)
        
        
        #mid
        
        x = self.mid_block1(x, t)
        ### 
        x = self.mid_spatial_attn1(x)
        x = self.mid_cross_attn1(x, kv=cond)
        x = self.mid_temporal_attn1(x, t)
        ###
        x = self.mid_spatial_attn2(x)
        x = self.mid_cross_attn2(x, kv=cond)
        x = self.mid_temporal_attn2(x, t)
        ###
        x = self.mid_spatial_attn3(x)
        x = self.mid_cross_attn3(x, kv=cond)
        x = self.mid_temporal_attn3(x, t)
        ###
        x = self.mid_spatial_attn4(x)
        x = self.mid_cross_attn4(x, kv=cond)
        x = self.mid_temporal_attn4(x, t)
        ###
        x = self.mid_block2(x, t)
        
        
        mid_res = self.mid_control(x)
        return down_res, mid_res
        

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            controlnet,
            *,
            image_size,
            num_frames,
            text_use_bert_cls=False,
            channels=3,
            timesteps=1000,
            loss_type='l1',
            use_dynamic_thres=False,  # from the Imagen paper
            dynamic_thres_percentile=0.9,
            ddim_timesteps=50,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.controlnet = controlnet

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.ddim_timesteps = ddim_timesteps

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile


    @torch.inference_mode()
    def p_sample_ddim(self, x, t, t_minus, indexes=None, cond=None, cond_ctrl=None, cond_scale=1., clip_denoised=True, controlnet_cond=None):
        b, *_, device = *x.shape, x.device
        if controlnet_cond is not None:
            down_additional_res_list, mid_additional_res = self.controlnet(x, t, cond = cond_ctrl, img_cond = controlnet_cond)
            x_recon = self.denoise_fn.forward_with_cond_scale(x, 
                                                              t, 
                                                              indexes=indexes, 
                                                              down_additional_res_list = down_additional_res_list,
                                                              mid_additional_res = mid_additional_res,
                                                              cond=cond, 
                                                              cond_scale=cond_scale)
        if cond_scale != 1:
            x_recon, x_recon_null = x_recon
            eps = get_eps_x_t(x_recon, x, t)
            eps_null = get_eps_x_t(x_recon_null, x, t)
            final_eps = eps_null + (eps - eps_null) * cond_scale
            x_recon = get_x0_x_t(final_eps, x, t)
        if t[0]<int(self.num_timesteps / self.ddim_timesteps):
            x = x_recon
        else:
            t_minus = torch.clip(t_minus, min=0.0)
            x = ddim_sample(x_recon, x, (t_minus * 1.0) / (self.num_timesteps), (t * 1.0) / (self.num_timesteps))
        return x

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_ctrl=None, cond_scale=1., controlnet_cond=None):

        device = cond.device
        bsz = shape[0]

        time_steps = range(0, self.num_timesteps+1, int(self.num_timesteps/self.ddim_timesteps))

        img = torch.randn(shape, device=device)
        indexes = []
        for b in range(bsz):
            index = np.arange(self.num_frames)
            indexes.append(torch.from_numpy(index))
        indexes = torch.stack(indexes, dim=0).long().to(device)
        for i, t in enumerate(tqdm(reversed(time_steps), desc='sampling loop time step',
                                   total=len(time_steps))):
            time = torch.full((bsz,), t, device=device, dtype=torch.float32)

            time_minus = time - int(self.num_timesteps / self.ddim_timesteps)
            img = self.p_sample_ddim(img, time, time_minus, indexes=indexes, cond=cond, cond_ctrl =cond_ctrl,
                                         cond_scale=cond_scale, controlnet_cond=controlnet_cond)
        return img

    @torch.inference_mode()
    def sample(self, cond=None, cond_ctrl=None, cond_scale=1., batch_size=16, controlnet_cond=None):
        device = next(self.denoise_fn.parameters()).device

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size[0], image_size[1]), cond=cond, cond_ctrl=cond_ctrl,
                                      cond_scale=cond_scale, controlnet_cond=controlnet_cond)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, indexes=None, cond=None, cond_ctrl=None, img_cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device

        x_noisy, noise = get_z_t(x_start, t)

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
        #     cond = cond.to(device)
        down_additional_res_list, mid_additional_res = self.controlnet(
                        x_noisy,
                        cond=cond_ctrl, 
                        time = t*self.num_timesteps,
                        img_cond = img_cond,
                        **kwargs       
                    )
        
        x_recon = self.denoise_fn(x_noisy, 
                                  t*self.num_timesteps, 
                                  indexes=indexes, 
                                  cond=cond,
                                  down_additional_res_list = down_additional_res_list,
                                  mid_additional_res = mid_additional_res,
                                  **kwargs)

        if self.loss_type == 'x0':
            loss = F.mse_loss(x_start.float(), x_recon.float())
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_start.float(), x_recon.float())
        # elif self.loss_type == 'v':
        #     v = get_v_x_t(x_recon, x_noisy, t)
        #     loss = F.mse_loss(v_real, v)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, setting_time=None, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        # check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size[0], w=img_size[1])
        if setting_time is None:
            t = torch.rand((b), device=device).float()
        else:
            t = setting_time
        return self.p_losses(x, t, *args, **kwargs)


# trainer class

CHANNELS_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}


def seek_all_images(img, channels=3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1


# tensor of shape (channels, frames, height, width) -> gif

def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    images = map(T.ToPILImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all=True, append_images=rest_imgs, duration=duration, loop=loop, optimize=optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(path, channels=3, transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels=channels)))
    return torch.stack(tensors, dim=1)


def identity(t, *args, **kwargs):
    return t


def normalize_img(t):
    return t * 2 - 1


def unnormalize_img(x_recon):
    x_recon = x_recon.clamp(-1, 1)
    return (x_recon + 1) * 0.5


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))




# trainer class

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            unet_path,
            vae,
            *,
            ema_decay=0.995,
            num_frames=16,
            train_batch_size=32,
            train_lr=1e-4,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            amp=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            num_sample_rows=4,
            max_grad_norm=None
    ):
        super().__init__()
        self.vae = vae
        self.model = diffusion_model
        # self.model.load_state_dict(torch.load("results/model-17.pt")['model'], strict=False)
        # self.ema = EMA(ema_decay)
        # self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        self.num_frames = diffusion_model.num_frames

        train_files = []
        # self.train_dataset = torch.load(data_path+'latents0.pt').squeeze(dim=1)
        # self.val_dataset = torch.load(data_path+'latents_val0.pt').squeeze(dim=1)
        self.train_dataset = get_data_loaders("MRI", 192, batch_size=self.batch_size, ifreturn_loader=False, ifexample = True, resize=False, age_normalize =True, only_control=False)
        self.dl = cycle(data.DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4))
        # self.dl_eval = data.DataLoader(self.val_dataset, batch_size=1, shuffle=True, pin_memory=True)
        self.opt = AdamW(self.model.controlnet.parameters(), lr=train_lr, betas=(0.9, 0.999), weight_decay=0.01)

        self.step = 0

        self.amp = amp
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        # self.reset_parameters()
        
        self.model.load_state_dict(load_file(unet_path), strict=False)
        print('loading ckpt from unet')
        self.model.controlnet.from_unet(self.model.denoise_fn) # load from unet 3d
        print('controlnet initialization completed!')
        if amp:
            mixed_precision = "fp16"
        else:
            mixed_precision = "fp32"
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulate_every,
            mixed_precision=mixed_precision,
            kwargs_handlers=[ddp_kwargs]
        )

        # train controlnet, not unet!
        
         # push it to device and frozen it 
        self.vae.to(self.accelerator.device)
        self.vae.requires_grad_(False)
        
        self.model.to(self.accelerator.device)
        self.model.requires_grad_(False)
        
        self.model.controlnet.requires_grad_(True)
        self.model.controlnet.train()
        
        self.model, self.dl, self.opt, self.step = self.accelerator.prepare(
            self.model, self.dl, self.opt, self.step
        )
        

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        self.accelerator.save_state(str(self.results_folder / f'final_ckpt_4'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            dirs = os.listdir(self.results_folder)
            dirs = [d for d in dirs if d.endswith("ckpt_3")]
            # dirs = sorted(dirs, key=lambda x: int(x.split("_")[0]))
            path = dirs[-1]

        self.step = 50000

        self.accelerator.load_state(os.path.join(self.results_folder, path), strict=False)

    def train(
            self,
            prob_focus_present=0.,
            focus_present_mask=None,
            log_fn=noop
    ):
        assert callable(log_fn)
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                with self.accelerator.accumulate(self.model):
                    
                    batch = next(self.dl)
                    # img_batch, seg_batch, seg_mask_name, age = batch
                    img_batch, seg_batch, seg_mask_name, age, sex, v_f, v_p, v_i, v_total = batch
                    # print(sex)
                    img = img_batch['pixel_values']
                    mask_condition = seg_batch
                    if sex == ['M']:
                        sex = -1
                    elif sex == ['F']:
                        sex = 1
                    else:
                        raise ValueError("Unexpected value in sex list")
                    # mask_condition = img_batch['condition_pixel_values'] # segmenation mask from synthseg
                    filename = img_batch['path']
                    img = img.to(self.accelerator.device,dtype = self.vae.dtype)
                    img = self.vae.encode(img).sample()
                    img = img * 0.188
                    # f = img.shape[2]
                    # img = self.vae.encode(img).sample()
                    # img = img * 0.19
                    # if self.amp:
                    #     img = img.to(self.accelerator.device,dtype=torch.float16)
                    #     mask_condition = mask_condition.to(self.accelerator.device,dtype=torch.float16)
                    # else:
                    #     img = img.to(self.accelerator.device,dtype=torch.float32)
                    #     mask_condition = mask_condition.to(self.accelerator.device,dtype=torch.float32)
                        
                    age_emb = age * torch.ones(img.shape[0], 250, 768)
                    sex_emb = sex * torch.ones(img.shape[0], 250, 768) 
                    non_emb = torch.ones(img.shape[0], 500, 768) 
                    non_emb =  non_emb.to(dtype=torch.float32, device=self.accelerator.device)
                    condition_emb = torch.concat((age_emb, sex_emb), dim=1)
                    condition_emb = condition_emb.to(dtype=torch.float32, device=self.accelerator.device)
                    # age_emb = age * torch.ones(img.shape[0], 250, 768)
                    # age_emb = age_emb.to(dtype=torch.float32, device=self.accelerator.device)

                    B, C, D, H, W = img.shape
                    
                    mask_condition = mask_condition.to(self.accelerator.device,dtype=img.dtype)
                    batch_images_inputs = img
                    
                    # extract res list
                    
                    # t = torch.rand((B), device=self.accelerator.device).float()
                    
                    # down_additional_res_list, mid_additional_res = self.controlnet(
                    #     batch_images_inputs,
                    #     cond=text,
                    #     time = t*1000,
                    #     img_cond = mask_condition,
                    #     prob_focus_present=prob_focus_present,
                    #     focus_present_mask=focus_present_mask,
                        
                    # )
                    # print(indexes.dtype, batch_images_inputs.dtype)
                    # TODO: training is for the controlnet, not the original unet.
                    # it is like we use the controlnet as an encoder, 1st we use controlnet to extract the feature of the seg mask
                    # and then we use the original trained sd decoder to give a x_0 prediction.
                    loss = self.model(
                        batch_images_inputs,
                        cond=condition_emb,
                        cond_ctrl=non_emb,
                        img_cond = mask_condition,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask
                    )
                    
                    self.accelerator.backward(loss)

                    if self.accelerator.sync_gradients:
                        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean()
                        print(f'{self.step}: {avg_loss},  {grad_norm}')
                        log = {'loss': avg_loss, 'grad_norm': grad_norm}

                self.opt.step()
                self.opt.zero_grad()

                if self.accelerator.sync_gradients:
                    # if self.step % self.update_ema_every == 0:
                    #     self.step_ema()
                    
                    
                    # inference mode:
                    # visualize the contion mask, corresponding image and ..
                    with torch.no_grad():
                        if self.step != 0 and self.step % (self.save_and_sample_every) == 0:
                            milestone = self.step // self.save_and_sample_every

                            self.save(milestone)
                            file_name = f"samples_epoch_{self.step}"

                            num_samples = self.num_sample_rows ** 2
                            batches = num_to_groups(num_samples, self.batch_size)
                            
                            mask_video_list = mask_condition
                            
                            one_gif = rearrange(mask_video_list, '(i j) c f h w -> c f (i h) (j w)',
                                                i=self.num_sample_rows)
                            video_path = str(self.results_folder / str(f'{str(milestone)}_mask_{seg_mask_name}_{file_name}.gif'))
                            video_tensor_to_gif(one_gif, video_path)
                            
                            if hasattr(self.model, 'module'):
                                all_videos_list = list(
                                    map(lambda n: self.model.module.sample(batch_size=n, cond=condition_emb, cond_ctrl=non_emb, controlnet_cond=mask_condition), batches))
                            else:
                                all_videos_list = list(
                                    map(lambda n: self.model.sample(batch_size=n, cond=condition_emb, cond_ctrl=non_emb, controlnet_cond=mask_condition), batches))
                            all_videos_list = torch.cat(all_videos_list, dim=0)
                            # all_videos_list = (img+1.0)/2.0
                            # all_videos_list, all_videos_list_lobe, all_videos_list_airway, all_videos_list_vessel = all_videos_list.chunk(4, dim=1)
                            # all_videos_list = torch.cat([all_videos_list, all_videos_list_lobe, all_videos_list_airway, all_videos_list_vessel], dim=0)
                            # path_nii = str(self.results_folder / str(f'{str(milestone)}_{file_name}.nii'))
                            # save_nii(all_videos_list[0].unsqueeze(dim=0), self.vae, path_nii)
                            all_videos_list = decode_from_latent(all_videos_list[0].unsqueeze(dim=0), self.vae)
                            all_videos_list = F.pad(all_videos_list, (2, 2, 2, 2))
        
                            one_gif = rearrange(all_videos_list, '(i j) c f h w -> c f (i h) (j w)',
                                                i=self.num_sample_rows)
                            video_path = str(self.results_folder / str(f'{str(milestone)}_{file_name}_condition{seg_mask_name}.gif'))
                            video_tensor_to_gif(one_gif, video_path)
                            log = {**log, 'sample': video_path}

                        # log_fn(log)
                        self.step += 1

        print('training completed')

model = Unet3D(
    dim=256,
    cond_dim=768,
    dim_mults=(1, 2, 4),
    #(batch,4,144,22,18)
    channels=4,
    attn_heads=8,
    attn_dim_head=32,
    use_bert_text_cond=False,
    init_dim=None,
    init_kernel_size=7,
    use_sparse_linear_attn=True,
    block_type='resnet',
    resnet_groups=8
)

controlnet = Controlnet3D(
    dim=256,
    cond_dim=768,
    dim_mults=(1, 2, 4),
    #(batch,4,144,22,18)
    channels=4,
    attn_heads=8,
    attn_dim_head=32,
    use_bert_text_cond=False,
    init_dim=None,
    init_kernel_size=7,
    use_sparse_linear_attn=True,
    block_type='resnet',
    resnet_groups=8
)

diffusion_model = GaussianDiffusion(
    denoise_fn=model,
    controlnet = controlnet,
    image_size=(24, 20),
    num_frames=40,
    text_use_bert_cls=False,
    channels=4,
    timesteps=1000,
    loss_type='x0',
    use_dynamic_thres=False,  # from the Imagen paper
    dynamic_thres_percentile=0.995,
    ddim_timesteps=50,
)



checkpoint_path = '/home/wepeng/binx/brain_gen/Open-Sora-Plan/recon_mask/checkpoints/vaenew_1_model_epoch_3.pth'
#load model
vae = CausalVAEModel()
vae.load_state_dict(torch.load(checkpoint_path))
vae.requires_grad_(False)

unet_path = '/home/wepeng/binx/brain_gen/Open-Sora-Plan/results_sdvae_192_age_sex/final_ckpt_2/model_1.safetensors'

trainer = Trainer(diffusion_model=diffusion_model,
                #   folder="/ocean/projects/asc170022p/yanwuxu/diffusion/data/R3/img_segs_128",
                  ema_decay=0.995,
                  vae=vae,
                  unet_path = unet_path,
                  num_frames=40,
                  train_batch_size=1,
                  train_lr=1e-4,
                  train_num_steps=1000000,
                  gradient_accumulate_every=4,
                  amp=True,
                  step_start_ema=10000,
                  update_ema_every=2,
                  save_and_sample_every=1000,
                  results_folder='./result_controlnet_444_after_padding',
                  num_sample_rows=1,
                  max_grad_norm=1.0)

trainer.load(-1)
trainer.train()