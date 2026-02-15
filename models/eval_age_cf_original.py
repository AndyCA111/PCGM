import sys
sys.path.append(".")
import math
import copy
import random
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
# from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from tqdm import tqdm
from einops import rearrange
# from dataloader import cache_transformed_train_data
import glob, os
from einops_exts import check_shape, rearrange_many
from rotary_embedding_torch import RotaryEmbedding
from dataloader_ad import get_data_loaders
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import nibabel as nib
import xformers, xformers.ops
import pandas as pd
# import sys



def save_nii(latents, model, save_path):
    latents = (latents /0.19).to(model.dtype)
    recon_img = model.decode(latents)
    recon_img = recon_img.permute(0,1,4,3,2)
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

# def ddim_inv_step(x0, x_t, t_p1, t):
#     # epsilon = get_eps_x_t(x_0, x_t, t) #noise on
#     alpha_cum_t = get_alpha_cum(t)[:, None, None, None, None]
#     alpha_cum_tp1 = get_alpha_cum(t_p1)[:, None, None, None, None]
#     x_tp1 = torch.sqrt(alpha_cum_tp1)*x0 + torch.sqrt((1-alpha_cum_tp1)/(1-alpha_cum_t)) * (x_t - alpha_cum_t*x0)
#     return x_tp1

def ddim_inv_step(x_0, x_t, t_p1, t):
    # epsilon = get_eps_x_t(x_0, x_t, t) #noise on
    epsilon = get_eps_x_t(x_0, x_t, t)
    alpha_cum = get_alpha_cum(t_p1)[:, None, None, None, None]
    x_tp1 = torch.sqrt(alpha_cum)*x_0 + torch.sqrt(1-alpha_cum)*epsilon
    return x_tp1

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


# relative positional bias

class RelativePositionBias(nn.Module):
    def __init__(
            self,
            heads=8,
            num_buckets=16,
            max_distance=128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')


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
        self.null_cond_emb = nn.Parameter(torch.randn(1, 500, cond_dim)) if self.has_cond else None

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
            null_cond_prob=0.1,
            focus_present_mask=None,
            prob_focus_present=0.
            # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), 'cond must be passed in if cond_dim specified'

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

        for block1, upsample, block2, temporal_block in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = upsample(x)
            x = block2(x, t)
            x = temporal_block(x, t)

        x = torch.cat((x, r), dim=1)
        x = self.final_conv_0(x, t)
        x = self.final_conv_1(x)
        return x


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
            ddim_inv_timesteps=80,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        self.ddim_timesteps = ddim_timesteps
        self.ddim_inv_timesteps = ddim_inv_timesteps
        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile


    @torch.inference_mode()
    def p_sample_ddim(self, x, t, t_minus, indexes=None, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device

        x_recon = self.denoise_fn.forward_with_cond_scale(x, t, indexes=indexes, cond=cond, cond_scale=cond_scale)
        if cond_scale != 1:
            x_recon, x_recon_null = x_recon
            eps = get_eps_x_t(x_recon, x, t)
            eps_null = get_eps_x_t(x_recon_null, x, t)
            final_eps = eps_null + (eps - eps_null) * cond_scale
            x_recon = get_x0_x_t(final_eps, x, t)
        if t[0]< int(self.num_timesteps / self.ddim_timesteps):
            x = x_recon
        else:
            t_minus = torch.clip(t_minus, min=0.0)
            x = ddim_sample(x_recon, x, (t_minus * 1.0) / (self.num_timesteps), (t * 1.0) / (self.num_timesteps))
        return x

    @torch.inference_mode()
    def p_sample_ddim_inv(self, x, t, t_plus, indexes=None, cond=None, cond_scale=1., clip_denoised=True):
        b, *_, device = *x.shape, x.device
        
        x_recon = self.denoise_fn.forward_with_cond_scale(x, t, indexes=indexes, cond=cond, cond_scale=cond_scale)
        if cond_scale != 1:
            x_recon, x_recon_null = x_recon
            eps = get_eps_x_t(x_recon, x, t)
            eps_null = get_eps_x_t(x_recon_null, x, t)
            final_eps = eps_null + (eps - eps_null) * cond_scale
            x_recon = get_x0_x_t(final_eps, x, t)
        if t[0] <= int(self.num_timesteps / self.ddim_inv_timesteps):
            x = x
        else:
            t_plus = torch.clip(t_plus, max=1000)
            x = ddim_inv_step(x_recon, x, (t_plus * 1.0) / (self.num_timesteps), (t * 1.0) / (self.num_timesteps))
        return x
    
    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.,startpoint=None):

        device = cond.device
        bsz = shape[0]

        time_steps = range(0, self.num_timesteps+1, int(self.num_timesteps/self.ddim_timesteps))
        # time_steps = list(time_steps)[1:] 
        if startpoint is not None:
            img = startpoint
        else:
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
            img = self.p_sample_ddim(img, time, time_minus, indexes=indexes, cond=cond,
                                         cond_scale=cond_scale)
        return img

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1., batch_size=16, startpoint=None):
        device = next(self.denoise_fn.parameters()).device

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size[0], image_size[1]), cond=cond,
                                      cond_scale=cond_scale, startpoint=startpoint)

    @torch.inference_mode()
    def ddim_inv_sample_loop(self, latents, shape, cond=None, cond_scale=1.):

        device = cond.device
        bsz = shape[0]

        time_steps = range(0, self.num_timesteps+1, int(self.num_timesteps/self.ddim_inv_timesteps))
        # time_steps = list(time_steps)[1:] 
        # img = torch.randn(shape, device=device)
        img = latents.clone()
        indexes = []
        for b in range(bsz):
            index = np.arange(self.num_frames)
            indexes.append(torch.from_numpy(index))
        indexes = torch.stack(indexes, dim=0).long().to(device)
        
        intermediate_latents = []
        for i, t in enumerate(tqdm(time_steps, desc='sampling loop time step',
                                   total=len(time_steps))):
            time = torch.full((bsz,), t, device=device, dtype=torch.float32)

            time_plus = time + int(self.num_timesteps / self.ddim_inv_timesteps)
            #get xt from xt-1
            img = self.p_sample_ddim_inv(img, time, time_plus, indexes=indexes, cond=cond,
                                         cond_scale=cond_scale)
            intermediate_latents.append(img)
        return torch.cat(intermediate_latents)




    @torch.inference_mode()
    def ddim_inv_sample(self, latents, new_cond=None, cond_scale=1., batch_size=16):
        
        device = next(self.denoise_fn.parameters()).device

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond)).to(device)

        batch_size = new_cond.shape[0] if exists(new_cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.ddim_inv_sample_loop(latents,(batch_size, channels, num_frames, image_size[0], image_size[1]), cond=new_cond,
                                      cond_scale=cond_scale)
    
    
    
    
    
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

    def p_losses(self, x_start, t, indexes=None, cond=None, noise=None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device

        x_noisy, noise = get_z_t(x_start, t)

        # if is_list_str(cond):
        #     cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
        #     cond = cond.to(device)

        x_recon = self.denoise_fn(x_noisy, t*self.num_timesteps, indexes=indexes, cond=cond, **kwargs)

        if self.loss_type == 'x0':
            loss = F.mse_loss(x_start, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_start, x_recon)
        # elif self.loss_type == 'v':
        #     v = get_v_x_t(x_recon, x_noisy, t)
        #     loss = F.mse_loss(v_real, v)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        # check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=img_size[0], w=img_size[1])
        t = torch.rand((b), device=device).float()
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


class Dataset(data.Dataset):
    def __init__(
            self,
            folder,
            image_size,
            channels=3,
            num_frames=16,
            horizontal_flip=False,
            force_num_frames=True,
            exts=['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames=num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform=self.transform)
        return self.cast_num_frames_fn(tensor)


# trainer class

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
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
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
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
        # self.train_dataset = get_data_loaders("MRI", 192, batch_size=self.batch_size, ifreturn_loader=False, ifexample = False, resize=False)
        self.train_dataset = get_data_loaders("MRI", 192, batch_size=self.batch_size, ifreturn_loader=False, ifexample = True, resize=False, age_normalize =True)
        self.dl = cycle(data.DataLoader(self.train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=4))
        # self.dl_eval = data.DataLoader(self.val_dataset, batch_size=1, shuffle=True, pin_memory=True)
        self.opt = AdamW(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), weight_decay=0.01)

        self.step = 0

        self.amp = amp
        self.max_grad_norm = max_grad_norm

        self.num_sample_rows = num_sample_rows
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

        if amp:
            mixed_precision = "fp16"
        else:
            mixed_precision = "fp32"

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulate_every,
            mixed_precision=mixed_precision,
        )

        self.model, self.ema_model, self.dl, self.opt, self.step, self.vae = self.accelerator.prepare(
            self.model, self.ema_model, self.dl, self.opt, self.step, self.vae
        )
        

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        self.accelerator.save_state(str(self.results_folder / f'{milestone}_ckpt'))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            dirs = os.listdir(self.results_folder)
            dirs = [d for d in dirs if d.endswith("ckpt_2")]
            # dirs = sorted(dirs, key=lambda x: int(x.split("_")[0]))
            path = dirs[-1]

        self.step = 50000

        self.accelerator.load_state(os.path.join(self.results_folder, path), strict=False)

    def sample(
            self,
            prob_focus_present=0.,
            focus_present_mask=None,
            sample_num = 30,
            log_fn=noop,
            start_step = 0,
    ):
        assert callable(log_fn)

        self.results_folder = os.path.join(str(self.results_folder), "test_sex_cf_sample_cfg_2_start_0")
        if not os.path.exists(self.results_folder) and self.accelerator.is_main_process:
            os.mkdir(self.results_folder)
        self.ema_model.eval()
        # self.ema_model.half()
        df = pd.read_csv('/home/wepeng/binx/brain_gen/Open-Sora-Plan/examples/subjects_cvpr.csv')
        for i, batch in tqdm(df.iterrows()):
            # ======================== get meta data : ========================
            new_age = batch['age']
            age_mean = 49.03
            age_std = 26.87
            sex = batch['sex']
            img_path = '/home/wepeng/data/MRI_3Set/' + batch['fname']
            # find max age brain:
            subject_id = None
            basic_brain_path = None
            fname = batch['fname']
            output_filename = fname.split('/', 1)[1]
            # Identify the basic brain based on 'fname' prefix
            subject_id = None
            basic_brain_path = None
            basic_brain_age = None
            basic_brain_sex = None

            # Identify the basic brain based on 'fname' prefix
            if fname.startswith("NCANDA"):
                # Extract the subject ID and construct baseline filename
                parts = fname.split('/')
                if len(parts) > 1:
                    subject_id = parts[1].split('_')[0] + '_' + parts[1].split('_')[1]  # e.g., NCANDA_S00051
                    baseline_fname = f"{subject_id}_baseline.nii.gz"
                    basic_brain_path = '/home/wepeng/data/MRI_3Set/' + '/'.join(parts[:-1]) + '/' + baseline_fname

                    # Find the row with the baseline path and get age, sex
                    baseline_row = df[df['fname'].str.contains(baseline_fname)]
                    if not baseline_row.empty:
                        basic_brain_age = baseline_row['age'].values[0]
                        basic_brain_sex = baseline_row['sex'].values[0]

            elif fname.startswith("ADNI_extract"):
                # Extract subject ID (e.g., "023_S_0058")
                parts = fname.split('_')
                if len(parts) > 2:
                    subject_id = f"{parts[1]}_{parts[2].split('-')[0]}_{parts[3].split('-')[0]}"
                print(subject_id)
                # Find all entries with the same subject ID and minimum age
                subject_rows = df[(df['fname'].str.startswith("ADNI_extract")) & (df['fname'].str.contains(subject_id))]
                if not subject_rows.empty:
                    min_age_row = subject_rows.loc[subject_rows['age'].idxmin()]  # Row with minimum age
                    basic_brain_path = '/home/wepeng/data/MRI_3Set/' + min_age_row['fname']
                    basic_brain_age = min_age_row['age']
                    basic_brain_sex = min_age_row['sex']
            # ======================== get basic brain : ========================
            print('basic_brain_path:', basic_brain_path)
            print('to_be_get_path:', output_filename)
            data = nib.load(basic_brain_path).get_fdata()
            max_value = np.percentile(data, 95)
            min_value = np.percentile(data, 5)
            data = np.where(data <= max_value, data, max_value)
            data = np.where(data <= min_value, 0., data)
            data = (data/max_value) * 2 - 1
            img2 = np.ones((160, 192, 157))*(-1)
            # img2 = np.ones((144, 192, 144))*(-1)
            img2[11:11+138,8:8+176,9:9+138] = data
            img = np.transpose(img2,(2,1,0))
            data = torch.from_numpy(img[None,:,:,:]).float()
            data = data.repeat(3, 1, 1, 1)
            img = data.unsqueeze(0)
            
            img = img.to(self.accelerator.device, dtype = self.vae.dtype)
            f = img.shape[2]
            img = self.vae.encode(img).sample()
            img = img * 0.19
            # ======================== get brain : ========================
            # diag = batch['diagnosis']
            if sex == 'M':
                sex = 0
                new_sex = 1
            elif sex == 'F':
                sex = 1
                new_sex = 0
            else:
                raise ValueError("Unexpected value in sex list")
            # if img_path == basic_brain_path:
            #     ori_path_nii =  os.path.join(str(self.results_folder), output_filename)
            #     save_nii(img, self.vae, ori_path_nii)
            #     continue
            # else:
            
            age = (basic_brain_age - age_mean) / age_std 
            if basic_brain_age< 40:
                new_age = (new_age - age_mean) / 24.15 
            else:
                new_age = (new_age - age_mean) / 24.15 
            age_emb = age * torch.ones(1, 250, 768)
            new_age_emb = new_age * torch.ones(1, 250, 768)
            sex_emb = sex * torch.ones(1, 250, 768) 
            new_sex_emb = new_sex * torch.ones(1, 250, 768) 
            condition_emb = torch.concat((age_emb, sex_emb), dim=1)
            new_condition_emb = torch.concat((new_age_emb, sex_emb), dim=1)
            condition_emb = condition_emb.to(dtype=torch.float32, device=self.accelerator.device) 
            new_condition_emb = new_condition_emb.to(dtype=torch.float32, device=self.accelerator.device) 
            with torch.no_grad():
                print(f'========= metadata: age: {new_age}, sex: {sex} ===============')
                num_samples = self.num_sample_rows ** 2
                batches = num_to_groups(num_samples, self.batch_size)
                # ori_path_nii =  os.path.join(str(self.results_folder), f'subject_{subj_name}_diag_{diag}.nii.gz')
                # save_nii(img, self.vae, ori_path_nii)
                # # for new_con in new_con_list:
                # # #new start point
                inv_latents = list(
                    map(lambda n: self.ema_model.module.ddim_inv_sample(batch_size=n, latents=img, new_cond=condition_emb, cond_scale=2), batches))
                inv_latents = torch.cat(inv_latents, dim=0)
                # print(inv_latents.shape)

                path_nii =  os.path.join(str(self.results_folder), output_filename)
                new_latents = list(
                    map(lambda n: self.ema_model.module.sample(batch_size=n, cond= new_condition_emb, cond_scale=2, startpoint=inv_latents[-(start_step + 1)][None]), batches))
                new_latents = torch.cat(new_latents, dim=0)
                save_nii(new_latents[0].unsqueeze(dim=0), self.vae, path_nii)
                    

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

diffusion_model = GaussianDiffusion(
    denoise_fn=model,
    image_size=(24, 20),
    num_frames=40,
    text_use_bert_cls=False,
    channels=4,
    timesteps=1000,
    loss_type='x0',
    use_dynamic_thres=False,  # from the Imagen paper
    dynamic_thres_percentile=0.995,
    ddim_inv_timesteps=80,
    ddim_timesteps=80,
)

checkpoint_path = '/home/wepeng/binx/brain_gen/Open-Sora-Plan/recon_mask/checkpoints/vaenew_1_model_epoch_3.pth'
#load model
vae = CausalVAEModel()
vae.load_state_dict(torch.load(checkpoint_path))
vae.requires_grad_(False)

trainer = Trainer(diffusion_model=diffusion_model,
                  ema_decay=0.995,
                  vae=vae,
                  num_frames=40,
                  train_batch_size=1,
                  train_lr=1e-4,
                  train_num_steps=1000000,
                  gradient_accumulate_every=8,
                  amp=True,
                  step_start_ema=10000,
                  update_ema_every=2,
                  save_and_sample_every=500,
                  results_folder='./results_sdvae_192_age_sex',
                  num_sample_rows=1,
                  max_grad_norm=1.0)

trainer.load(-1)
trainer.sample()