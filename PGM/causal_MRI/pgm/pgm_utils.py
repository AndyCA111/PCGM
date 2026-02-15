import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import copy
import os
import random
from typing import Dict, List, Optional

import imageio
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, nn

from .pgm_hps import Hparams


## Utils copied from the utils file in previous src
def seed_all(seed, deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def linear_warmup(warmup_iters):
    def f(iter):
        return 1.0 if iter > warmup_iters else iter / warmup_iters

    return f


def beta_anneal(beta, step, anneal_steps):
    return min(beta, (max(1e-11, step) / anneal_steps) ** 2)


def normalize(x, x_min=None, x_max=None, zero_one=False):
    if x_min is None:
        x_min = x.min()
    if x_max is None:
        x_max = x.max()
    print(f"max: {x_max}, min: {x_min}")
    x = (x - x_min) / (x_max - x_min)  # [0,1]
    return x if zero_one else 2 * x - 1  # else [-1,1]


def log_standardize(x):
    log_x = torch.log(x.clamp(min=1e-12))
    return (log_x - log_x.mean()) / log_x.std().clamp(min=1e-12)  # mean=0, std=1


def exists(val):
    return val is not None


def is_float_dtype(dtype):
    return any(
        [
            dtype == float_dtype
            for float_dtype in (
                torch.float64,
                torch.float32,
                torch.float16,
                torch.bfloat16,
            )
        ]
    )


def clamp(value, min_value=None, max_value=None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value


class EMA(nn.Module):
    """
    Adapted from: https://github.com/lucidrains/ema-pytorch/blob/main/ema_pytorch/ema_pytorch.py
    Implements exponential moving average shadowing for your model.

    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.

    @crowsonkb's notes on EMA Warmup:

    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).

    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """

    def __init__(
        self,
        model,
        beta=0.999,
        update_after_step=100,
        update_every=1,
        inv_gamma=1.0,
        power=1.0,
        min_value=0.0,
        param_or_buffer_names_no_ema=set(),
    ):
        super().__init__()
        self.beta = beta
        self.online_model = model

        try:
            self.ema_model = copy.deepcopy(model)
        except:
            print(
                "Your model was not copyable. Please make sure you are not using any LazyLinear"
            )
            exit()

        self.ema_model.requires_grad_(False)
        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = (
            param_or_buffer_names_no_ema  # parameter or buffer
        )

        self.register_buffer("initted", torch.Tensor([False]))
        self.register_buffer("step", torch.tensor([0]))

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def copy_params_from_model_to_ema(self):
        for ma_params, current_params in zip(
            list(self.ema_model.parameters()), list(self.online_model.parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            ma_params.data.copy_(current_params.data)

        for ma_buffers, current_buffers in zip(
            list(self.ema_model.buffers()), list(self.online_model.buffers())
        ):
            if not is_float_dtype(current_buffers.dtype):
                continue

            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value=0.0)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power

        if epoch <= 0:
            return 0.0

        return clamp(value, min_value=self.min_value, max_value=self.beta)

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.online_model)

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(
            list(current_model.named_parameters()), list(ma_model.named_parameters())
        ):
            if not is_float_dtype(current_params.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            difference = ma_params.data - current_params.data
            difference.mul_(1.0 - current_decay)
            ma_params.sub_(difference)

        for (name, current_buffer), (_, ma_buffer) in zip(
            list(current_model.named_buffers()), list(ma_model.named_buffers())
        ):
            if not is_float_dtype(current_buffer.dtype):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            difference = ma_buffer - current_buffer
            difference.mul_(1.0 - current_decay)
            ma_buffer.sub_(difference)

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)


def write_images(args: Hparams, model: nn.Module, batch: Dict[str, Tensor]):
# def write_images(args, model: nn.Module, batch: Dict[str, Tensor]):
    bs, c, h, w = batch["x"].shape
    # original imgs, channels last, [0,255]
    orig = (batch["x"].permute(0, 2, 3, 1) + 1.0) * 127.5
    orig = orig.detach().cpu().numpy().astype(np.uint8)
    viz_images = [orig]

    def postprocess(x: Tensor):
        x = (x.permute(0, 2, 3, 1) + 1.0) * 127.5  # channels last, [0,255]
        return x.detach().cpu().numpy()

    def pseudo_counterfactuals(
        model: nn.Module,
        z: List[Tensor],
        pa: Dict[str, Tensor],
        cf_pa: Dict[str, Tensor],
        x: Optional[Tensor] = None,
        alpha: Optional[float] = None,
        t: Optional[float] = None,
    ):
        """Note that this function is only here for debugging purposes.
        It does not take into account the associated causal graph nor infer x's
        (observation space) exogenous noise term "u". For a complete example of
        counterfactual inference you may refer to pgm/dscm.py or our demo:

          https://huggingface.co/spaces/mira-causality/counterfactuals/blob/main/app.py
          (specifically the counterfactual_inference() function).

        """
        # x = g(pa, z)
        x_rec, _ = model.forward_latents(latents=z, parents=pa, t=t)
        x_rec = postprocess(x_rec)

        # x* = g(pa*, z), direct effect counterfactual
        cf_x, _ = model.forward_latents(latents=z, parents=cf_pa, t=t)
        _x = postprocess(cf_x)
        viz_images.append(_x.astype(np.uint8))
        viz_images.append((_x - x_rec).astype(np.uint8))

        if model.cond_prior:
            cf_z = model.abduct(x=x, parents=pa, cf_parents=cf_pa, alpha=alpha, t=t)
            # alternative: z* ~ q(z* | x*, pa*)
            # cf_z = model.abduct(x=cf_x, parents=cf_pa)

            # x* = g(pa, z*), indirect effect counterfactual
            _x, _ = model.forward_latents(latents=cf_z, parents=pa, t=t)
            _x = postprocess(_x)
            viz_images.append(_x.astype(np.uint8))
            viz_images.append((_x - x_rec).astype(np.uint8))

            # x* = g(pa*, z*), total effect counterfactual
            _x, _ = model.forward_latents(latents=cf_z, parents=cf_pa, t=t)
            _x = postprocess(_x)
            viz_images.append(_x.astype(np.uint8))
            viz_images.append((_x - x_rec).astype(np.uint8))
        return

    # reconstructions, first abduct z from q(z|x,pa)
    zs = model.abduct(x=batch["x"], parents=batch["pa"])
    # print(len(zs), zs[0]['z'].keys())
    n_latents_viz = 0  # 0 for simple vae
    l_points = np.floor(np.linspace(0, 1, n_latents_viz + 2) * len(zs)).astype(int)[
        1:
    ]  # [1:-1]

    for l in l_points:
        # reconstruc using first l latent z's
        if model.cond_prior:
            z_l = [zs[i]["z"] for i in range(l)]
        else:
            z_l = zs[:l]
        x, _ = model.forward_latents(latents=z_l, parents=batch["pa"], t=0.1)
        x = postprocess(x)
        viz_images.append(x.astype(np.uint8))
    viz_images.append(orig * 0)

    # random samples at different temps
    for temp in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x, _ = model.sample(parents=batch["pa"], return_loc=True, t=temp)
        x = postprocess(x)
        viz_images.append(x.astype(np.uint8))

    # compute counterfactuals
    idx = np.arange(bs)
    rng = np.random.RandomState(1)
    rng.shuffle(idx)
    alpha, t = 0.6, 0.5

    # undo input res repetition of parents for compatbility with simple vae
    if args.expand_pa:
        _pa = batch["pa"][:, :, 0, 0].clone()
        assert len(_pa.shape) == 2
    else:
        _pa = batch["pa"].clone()

    for l in l_points:
        viz_images.append(orig * 0)  # empty row

        for ii in range(bs):
            # copy ith (x, pa), repeat it for num attribute we can intervene
            if model.cond_prior:
                x = copy.deepcopy(batch["x"][ii])
                x = x[None, ...].repeat(args.context_dim, 1, 1, 1)
            pa = copy.deepcopy(_pa[ii])
            pa = pa[None, ...].repeat(args.context_dim, 1)
            # intervening on each attribute separately
            cf_pa = pa.clone()

            # format interventional parents according to each dataset
            if "ukbb" in args.hps:
                if args.parents_x == [
                    "mri_seq",
                    "brain_volume",
                    "ventricle_volume",
                    "sex",
                ]:
                    assert args.context_dim == 4
                    cf_pa[0, 0] = 1 - cf_pa[0, 0]  # invert mri_seq
                    cf_pa[1, 1] = _pa[idx[ii], 1]  # random bvol intervention
                    cf_pa[2, 2] = _pa[idx[ii], 2]  # random vvol intervention
                    cf_pa[3, 3] = 1 - cf_pa[3, 3]  # invert sex
                elif args.parents_x == ["mri_seq", "brain_volume", "ventricle_volume"]:
                    assert args.context_dim == 3
                    cf_pa[0, 0] = 1 - cf_pa[0, 0]  # invert mri_seq
                    cf_pa[1, 1] = _pa[idx[ii], 1]  # random bvol intervention
                    cf_pa[2, 2] = _pa[idx[ii], 2]  # random vvol intervention
                else:
                    NotImplementedError(f"{args.parents_x} not configured.")

            elif "morphomnist" in args.hps:
                assert args.context_dim == 12
                cf_pa[0, 0] = _pa[idx[ii], 0]  # random thickness intervention
                cf_pa[1, 1] = _pa[idx[ii], 1]  # random intensity intervention
                cf_pa[2:, 2:] = torch.eye(10)  # intervention for each digit

            elif "cmnist" in args.hps:
                assert args.context_dim == 20
                cf_pa[:10, :10] = torch.eye(10)  # intervention for each digit
                cf_pa[10:, 10:] = torch.eye(10)  # intervention for each colour
            else:
                NotImplementedError

            # repeat conditioning by input res, used for HVAE parent concatenation
            if args.expand_pa:
                pa = pa[..., None, None].repeat(1, 1, *(args.input_res,) * 2)
                cf_pa = cf_pa[..., None, None].repeat(1, 1, *(args.input_res,) * 2)

            # resolves to (1) for simple vae or (1,1,1) for HVAE
            n_dims = (len(pa.shape) - 1) * (1,)

            # to get counterfactuals of each attribute using same z
            z_i = []
            for z in zs:
                if model.cond_prior:
                    assert type(z) is dict
                    z_dict = {}
                    for k, v in z.items():
                        z_dict[k] = v[ii].repeat(args.context_dim, *n_dims)
                    z_i.append(z_dict)
                else:
                    z_i.append(z[ii].repeat(args.context_dim, *n_dims))

            # for partial abduction of z, e.g. fix first l latent z's only
            if model.cond_prior:
                z_l = [z_i[j]["z"] for j in range(l)]
            else:
                z_l = z_i[:l]

            if model.cond_prior:
                pseudo_counterfactuals(model, z_l, pa, cf_pa, x=x, alpha=alpha, t=t)
            else:
                pseudo_counterfactuals(model, z_l, pa, cf_pa, t=t)
            viz_images.append(orig * 0)  # empty row

    # zero pad each row to have same number of columns for plotting
    for j, img in enumerate(viz_images):
        s = img.shape[0]
        if s < bs:
            pad = np.zeros((bs - s, *img.shape[1:])).astype(np.uint8)
            viz_images[j] = np.concatenate([img, pad], axis=0)
    # concat all images and save to disk
    n_rows = len(viz_images)
    im = (
        np.concatenate(viz_images, axis=0)
        .reshape((n_rows, bs, h, w, c))
        .transpose([0, 2, 1, 3, 4])
        .reshape([n_rows * h, bs * w, c])
    )
    imageio.imwrite(os.path.join(args.save_dir, f"viz-{args.iter}.png"), im)


# plt.rcParams['figure.facecolor'] = 'white'


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        v_ext = np.max([np.abs(self.vmin), np.abs(self.vmax)])
        x, y = [-v_ext, self.midpoint, v_ext], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def check_nan(input_dict):
    nans = 0
    for k, v in input_dict.items():
        k_nans = torch.isnan(v).sum()
        nans += k_nans
        if k_nans > 0:
            print(f'\nFound {k_nans} nan(s) in {k}, skipping step.')
    return nans


def update_stats(stats, elbo_fn):
    """Accumulate tracked summary statistics."""

    def _update(trace, dist='p'):
        for name, node in trace.nodes.items():
            if node['type'] == 'sample':
                k = 'log' + dist + '(' + name + ')'
                if k not in stats:
                    stats[k] = 0
                stats[k] += node['log_prob'].sum().item()
        return stats

    _update(elbo_fn.trace_storage['model'], dist='p')
    _update(elbo_fn.trace_storage['guide'], dist='q')
    return stats


def plot(x, fig=None, ax=None, nrows=1, cmap='Greys_r', norm=None, cbar=False, set_cbar_ticks=True, logger=None):
    m, n = nrows, x.shape[0] // nrows
    if ax is None:
        fig, ax = plt.subplots(m, n, figsize=(n * 4, 8))
    im = []
    for i in range(m):
        for j in range(n):
            idx = (i, j) if m > 1 else j
            ax = [ax] if n == 1 else ax
            _x = x[i * n + j].squeeze()
            if norm is not None:
                norm = MidpointNormalize(vmin=_x.min(), midpoint=0, vmax=_x.max())
                # norm = colors.TwoSlopeNorm(vmin=_x.min(), vcenter=0., vmax=_x.max())
            # logger.info(f"ax[idx] is: {type(ax[idx])}, m: {m}, n: {n}, shape: {np.shape(ax[idx])}")
            _im = ax[idx].imshow(_x, cmap=cmap, norm=norm)
            im.append(_im)
            ax[idx].axes.xaxis.set_ticks([])
            ax[idx].axes.yaxis.set_ticks([])

    # plt.tight_layout()

    if cbar:
        if fig:
            fig.subplots_adjust(wspace=-0.275, hspace=0.25)
        for i in range(m):
            for j in range(n):
                idx = [i, j] if m > 1 else j
                # cbar_ax = fig.add_axes([
                #     ax[idx].get_position().x0 + 0.0025, # left
                #     ax[idx].get_position().y1, # bottom
                #     0.003, # width
                #     ax[idx].get_position().height # height
                # ])
                cbar_ax = fig.add_axes([
                    ax[idx].get_position().x0,
                    ax[idx].get_position().y0 - 0.015,
                    ax[idx].get_position().width,
                    0.0075
                ])
                cbar = plt.colorbar(im[i * n + j], cax=cbar_ax,
                                    orientation="horizontal")  # , ticks=mticker.MultipleLocator(25)) #, ticks=mticker.AutoLocator())
                # cbar.ax.tick_params(rotation=0)
                # cbar.ax.locator_params(nbins=5)
                _x = x[i * n + j].squeeze()

                if set_cbar_ticks:
                    d = 20
                    _vmin, _vmax = _x.min().abs().item(), _x.max().item()
                    _vmin = -(_vmin - (_vmin % d))
                    _vmax = _vmax - (_vmax % d)
                    lt = [_vmin, 0, _vmax]

                    if (np.abs(_vmin) - 0) > d or (_vmax - 0) > d:
                        lt.insert(1, _vmin // 2)
                        lt.insert(-2, _vmax // 2)
                    cbar.set_ticks(lt)
                else:
                    cbar.ax.locator_params(nbins=5)
                    cbar.formatter.set_powerlimits((0, 0))

                cbar.outline.set_visible(False)
    return fig, ax


@torch.no_grad()
def plot_cf(x, cf_x, pa, cf_pa, do, var_cf_x=None, num_images=8, logger=None):
    n = num_images  # 8 columns
    x = (x[:n].detach().cpu() + 1) * 127.5
    cf_x = (cf_x[:n].detach().cpu() + 1) * 127.5
    # logger.info(f"x: {x.size()}")
    fs = 16  # font size
    m = 3 if var_cf_x is None else 4  # nrows
    s = 5
    fig, ax = plt.subplots(m, n, figsize=(n * s - 6, m * s))
    # fig, ax = plt.subplots(m, n, figsize=(n*s, m*s+2))
    # logger.info(f"ax: {np.shape(ax)}")
    # logger.info(f"ax[0]: {type(ax[0])} {np.shape(ax[0])}, m: {m}, s: {s}, n: {n}")
    _, _ = plot(x, ax=ax[0])
    _, _ = plot(cf_x, ax=ax[1])
    _, _ = plot(cf_x - x, ax=ax[2], fig=fig, cmap='RdBu_r', cbar=True,
                norm=MidpointNormalize(midpoint=0))
    if var_cf_x is not None:
        _, _ = plot(var_cf_x[:n].detach().sqrt().cpu(),
                    fig=fig, cmap='jet', ax=ax[3], cbar=True, set_cbar_ticks=False)

    sex_categories = ['male', 'female']  # 0,1
    race_categories = ['White', 'Asian', 'Black']  # 0,1,2
    finding_categories = ['No finding', 'Finding']

    for j in range(n):
        msg = ''
        for i, (k, v) in enumerate(do.items()):
            if k == 'sex':
                vv = sex_categories[int(v[j].item())]
                kk = 's'
            elif k == 'age':
                vv = str(v[j].item())
                kk = 'a'
            elif k == 'race':
                vv = race_categories[int(torch.argmax(v[j], dim=-1))]
                kk = 'r'
            elif k =='finding':
                vv = finding_categories[int(v[j].item())]
                kk = 'f'
            msg += kk + '{{=}}' + vv
            msg += ', ' if (i + 1) < len(list(do.keys())) else ''

        s = str(sex_categories[int(pa['sex'][j].item())])
        r = str(race_categories[int(torch.argmax(pa['race'][j], dim=-1))])
        a = str(int(pa['age'][j].item()))
        f = str(finding_categories[int(pa['finding'][j].item())])


        ax[0, j].set_title(f'a={a}, s={s}, \n r={r}, f={f}',
                           pad=8, fontsize=fs - 4, multialignment='center', linespacing=1.5)
        ax[1, j].set_title(f'do(${msg}$)', fontsize=fs, pad=10)

        # plot counterfactual
        cf_s = str(sex_categories[int(cf_pa['sex'][j].item())])
        cf_a = str(np.round(cf_pa['age'][j].item(), 1))
        cf_r = str(race_categories[int(torch.argmax(cf_pa['race'][j], dim=-1))])
        cf_f = str(finding_categories[int(cf_pa['finding'][j].item())])

        ax[1, j].set_xlabel(
            rf'$\widetilde{{a}}{{=}}{cf_a}, \ \widetilde{{s}}{{=}}{cf_s}, \ \widetilde{{r}}{{=}}{cf_r},  \ \widetilde{{f}}{{=}}{cf_f}$',
            labelpad=9, fontsize=fs - 4, multialignment='center', linespacing=1.25)

    ax[0, 0].set_ylabel('Observation', fontsize=fs + 2, labelpad=8)
    ax[1, 0].set_ylabel('Counterfactual', fontsize=fs + 2, labelpad=8)
    ax[2, 0].set_ylabel('Treatment Effect', fontsize=fs + 2, labelpad=8)
    if var_cf_x is not None:
        ax[3, 0].set_ylabel('Uncertainty', fontsize=fs + 2, labelpad=8)
    return fig

def calculate_loss(pred_batch, target_batch, loss_norm="l1", soft_loss="BCElogits"):
    "Calculate the losses for pred_bacth"
    loss=0
    for k in pred_batch.keys():
        assert pred_batch[k].size()==target_batch[k].size(), f"{k} size does not match, pred_batch size {pred_batch[k].size()}; target batch size {target_batch[k].size()}"
        if k=="age":
            if loss_norm=="l1":
                loss+=torch.nn.L1Loss()(pred_batch[k], target_batch[k]) 
            elif loss_norm=="l2":
                loss+=torch.nn.MSELoss()(pred_batch[k], target_batch[k]) 
        elif k in ["sex", "finding"]:
            if soft_loss=="BCElogits":
                loss+=torch.nn.BCEWithLogitsLoss()(pred_batch[k], target_batch[k])
            elif soft_loss=="l1":
                loss+=torch.nn.L1Loss()(pred_batch[k], target_batch[k]) 
        elif k=="race":
            if soft_loss=="BCElogits":
                loss+=torch.nn.CrossEntropyLoss()(pred_batch[k], target_batch[k])
            elif soft_loss=="l1":
                loss+=torch.nn.L1Loss()(pred_batch[k], target_batch[k]) 
    return loss



@torch.no_grad()
def plot_cf_with_original_model(x, cf_x, pa, cf_pa, do, cf_x_orig, var_cf_x=None, var_cf_x_original=None, num_images=8, logger=None):
    n = num_images  # 8 columns
    x = (x[:n].detach().cpu() + 1) * 127.5
    cf_x_orig = (cf_x_orig[:n].detach().cpu() + 1) * 127.5
    cf_x = (cf_x[:n].detach().cpu() + 1) * 127.5
    # logger.info(f"x: {x.size()}")
    fs = 16  # font size
    m = 5 if var_cf_x is None else 7  # nrows
    s = 5
    fig, ax = plt.subplots(m, n, figsize=(n * s - 6, m * s))
    fig.subplots_adjust(wspace=-0.1, hspace=1.0)
    _, _ = plot(x, ax=ax[0])
    _, _ = plot(cf_x_orig, ax=ax[1]) # Counterfactuals before training
    _, _ = plot(cf_x_orig - x,  ax=ax[2], fig=fig, cmap='RdBu_r', cbar=True,
                norm=MidpointNormalize(midpoint=0)) # Treatment effect before training
    _, _ = plot(var_cf_x_original[:n].detach().sqrt().cpu(),
                    fig=fig, cmap='jet', ax=ax[3], cbar=True, set_cbar_ticks=False)
    
    _, _ = plot(cf_x, ax=ax[4]) # Counterfactuals after training
    _, _ = plot(cf_x - x, ax=ax[5], fig=fig, cmap='RdBu_r', cbar=True,
                norm=MidpointNormalize(midpoint=0)) # Treatment effect after training
    _, _ = plot(var_cf_x[:n].detach().sqrt().cpu(),
                    fig=fig, cmap='jet', ax=ax[6], cbar=True, set_cbar_ticks=False)
    

    sex_categories = ['male', 'female']  # 0,1
    race_categories = ['White', 'Asian', 'Black']  # 0,1,2
    finding_categories = ['No finding', 'Finding']

    for j in range(n):
        msg = ''
        for i, (k, v) in enumerate(do.items()):
            if k == 'sex':
                vv = sex_categories[int(v[j].item())]
                kk = 's'
            elif k == 'age':
                vv = str(v[j].item())
                kk = 'a'
            elif k == 'race':
                vv = race_categories[int(torch.argmax(v[j], dim=-1))]
                kk = 'r'
            elif k =='finding':
                vv = finding_categories[int(v[j].item())]
                kk = 'f'
            msg += kk + '{{=}}' + vv
            msg += ', ' if (i + 1) < len(list(do.keys())) else ''

        s = str(sex_categories[int(pa['sex'][j].item())])
        r = str(race_categories[int(torch.argmax(pa['race'][j], dim=-1))])
        a = str(int(pa['age'][j].item()))
        f = str(finding_categories[int(pa['finding'][j].item())])


        ax[0, j].set_title(f'a={a}, s={s}, \n r={r}, f={f}',
                           pad=8, fontsize=fs - 4, multialignment='center', linespacing=1.5)
        ax[1, j].set_title(f'do(${msg}$)', fontsize=fs, pad=10)

        # plot counterfactual
        cf_s = str(sex_categories[int(cf_pa['sex'][j].item())])
        cf_a = str(np.round(cf_pa['age'][j].item(), 1))
        cf_r = str(race_categories[int(torch.argmax(cf_pa['race'][j], dim=-1))])

        ax[1, j].set_xlabel(
        # ax[2, j].set_title(
            rf'$\widetilde{{a}}{{=}}{cf_a}, \ \widetilde{{s}}{{=}}{cf_s}, \ \widetilde{{r}}{{=}}{cf_r}$',
            labelpad=9, fontsize=fs - 4, multialignment='center', linespacing=1.25)
        
        # ax[3, j].set_title(f'do(${msg}$)', fontsize=fs, pad=20)
        ax[3, j].set_xlabel(
            rf'$\widetilde{{a}}{{=}}{cf_a}, \ \widetilde{{s}}{{=}}{cf_s}, \ \widetilde{{r}}{{=}}{cf_r}$',
            labelpad=9, fontsize=fs - 4, multialignment='center', linespacing=1.25)

    ax[0, 0].set_ylabel('Observation', fontsize=fs + 2, labelpad=8)
    ax[1, 0].set_ylabel('Counterfactual', fontsize=fs + 2, labelpad=8)
    ax[2, 0].set_ylabel('Treatment Effect', fontsize=fs + 2, labelpad=8)
    ax[3, 0].set_ylabel('Uncertainty', fontsize=fs + 2, labelpad=8)

    ax[4, 0].set_ylabel('Counterfactual', fontsize=fs + 2, labelpad=8)
    ax[5, 0].set_ylabel('Treatment Effect', fontsize=fs + 2, labelpad=8)
    ax[6, 0].set_ylabel('Uncertainty', fontsize=fs + 2, labelpad=8)
    return fig