import logging
from typing import Callable, Optional

import torch
from torchdiffeq import odeint
import torch.nn as nn
log = logging.getLogger()

import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np
import math


def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return stopgrad(w) * loss


def cosine_annealing(start, end, step, total_steps):
    cos_inner = math.pi * step / total_steps
    return end + 0.5 * (start - end) * (1 + math.cos(cos_inner))


## partially from https://github.com/haidog-yaqub/MeanFlow
class MeanFlow():
    def __init__(
        self, 
        steps=1,  
        flow_ratio=0.75,
        time_dist=['lognorm', -0.4, 1.0],
        w=0.3,
        k=0.9,
        cfg_uncond='u',
        jvp_api='autograd',
    ):
        super().__init__()
        self.flow_ratio = flow_ratio
        self.time_dist = time_dist
        self.w = w
        self.k = k
        self.steps = steps
        
        self.cfg_uncond = cfg_uncond
        self.jvp_api = jvp_api
        assert jvp_api in ['funtorch', 'autograd'], "jvp_api must be 'funtorch' or 'autograd'"
        if jvp_api == 'funtorch':
            self.jvp_fn = torch.func.jvp
            self.create_graph = False
        elif jvp_api == 'autograd':
            self.jvp_fn = torch.autograd.functional.jvp
            self.create_graph = True
        log.info(f'MeanFlow initialized with {steps} steps')

    def sample_t_r(self, batch_size, device):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))  

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        # we don't use self.flow ratio if we use scheduler
        # !TODO: implement flow ratio scheduler
        num_selected = int(self.flow_ratio * batch_size)  
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)
        return t, r
    
    def to_prior(self, fn: Callable, x1: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x1)

    @torch.no_grad()
    def to_data(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        return self.run_t0_to_t1(fn, x0)
        
    def run_t0_to_t1(self, fn: Callable, x0: torch.Tensor) -> torch.Tensor:
        t = torch.ones((x0.shape[0],), device=x0.device,dtype=x0.dtype)
        r = torch.zeros((x0.shape[0],), device=x0.device,dtype=x0.dtype)
        steps = torch.linspace(1, 0, self.steps + 1).to(device=x0.device,dtype=x0.dtype)
        for ti, t in enumerate(steps[:-1]):
            t = t.expand(x0.shape[0])
            next_t = steps[ti + 1].expand(x0.shape[0])
            u_flow = fn(t=t, r=next_t, x=x0)
            dt = (t - next_t).mean()
            x0 = x0 - dt * u_flow
        return x0

    def loss(self,
            fn: Callable,  
            x0: torch.Tensor,
            text_f: torch.Tensor,
            text_f_c: torch.Tensor,
            text_f_undrop: torch.Tensor,
            text_f_c_undrop: torch.Tensor,
            empty_string_feat: torch.Tensor,
            empty_string_feat_c: torch.Tensor):
        if isinstance(fn, torch.nn.parallel.DistributedDataParallel):
            fn = fn.module
        batch_size = x0.shape[0]
        device = x0.device
        e = torch.randn_like(x0)
        t, r = self.sample_t_r(batch_size, device)
        t_ = rearrange(t, "b -> b 1 1 ")
        r_ = rearrange(r, "b -> b 1 1 ")
        z = (1 - t_) * x0 + t_ * e  # r < t
        v = e - x0
        
        if self.w is not None:
            u_text_f = empty_string_feat.expand(batch_size, -1, -1)
            u_text_f_c = empty_string_feat_c.expand(batch_size, -1)
            u_t = fn(latent=z, 
                     text_f=u_text_f,
                     text_f_c=u_text_f_c,
                     r=t,
                     t=t).detach().requires_grad_(False)
            u_t_c = fn(latent=z, 
                       text_f=text_f_undrop,
                       text_f_c=text_f_c_undrop,
                       r=t,
                       t=t).detach().requires_grad_(False)
        
            v_hat = self.w * v + self.k * u_t_c + (1 - self.w - self.k) * u_t
        else:
            v_hat = v

        device = z.device
        model_partial = partial(fn, text_f=text_f,text_f_c=text_f_c)
        jvp_args = (
            lambda z_f, r_f, t_f: model_partial(latent=z_f, r=r_f, t=t_f),
            (z, r, t),
            (v_hat, torch.zeros_like(r), torch.ones_like(t)),
        )
        if self.create_graph:
            u, dudt = self.jvp_fn(*jvp_args, create_graph=True)
        else:
            u, dudt = self.jvp_fn(*jvp_args)
        u_tgt = v_hat - (t_ - r_) * dudt
        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        return loss, r, t


if __name__ == '__main__': 
    pass