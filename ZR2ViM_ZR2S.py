import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


def q_shift(input, shift_pixel=1, gamma=1 / 4):
    assert gamma <= 1 / 4
    B, C, H, W = input.shape
    output = torch.zeros_like(input)
    output[:, 0:int(C * gamma), :, shift_pixel:W] = input[:, 0:int(C * gamma), :, 0:W - shift_pixel]
    output[:, int(C * gamma):int(C * gamma * 2), :, 0:W - shift_pixel] = input[:, int(C * gamma):int(C * gamma * 2), :,
                                                                        shift_pixel:W]
    output[:, int(C * gamma * 2):int(C * gamma * 3), shift_pixel:H, :] = input[:, int(C * gamma * 2):int(C * gamma * 3),
                                                                        0:H - shift_pixel, :]
    output[:, int(C * gamma * 3):int(C * gamma * 4), 0:H - shift_pixel, :] = input[:,
                                                                            int(C * gamma * 3):int(C * gamma * 4),
                                                                            shift_pixel:H, :]
    output[:, int(C * gamma * 4):, ...] = input[:, int(C * gamma * 4):, ...]
    return output


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C

        w = w.float()
        u = u.float()
        k = k.float()
        v = v.float()

        y = torch.zeros((B, T, C), device=w.device, dtype=torch.float)

        for b in range(B):
            for c in range(C):
                state = torch.zeros(1, device=w.device, dtype=torch.float)
                num = torch.zeros(1, device=w.device, dtype=torch.float)
                den = torch.zeros(1, device=w.device, dtype=torch.float)

                for t in range(T):
                    kt = k[b, t, c]
                    vt = v[b, t, c]
                    wt = w[c]
                    ut = u[c]

                    num = num * wt + kt * vt
                    den = den * wt + kt
                    state = ut * state + num / (den + 1e-6)
                    y[b, t, c] = state

        ctx.save_for_backward(w, u, k, v, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        w, u, k, v, y = ctx.saved_tensors

        grad_w = torch.zeros_like(w)
        grad_u = torch.zeros_like(u)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        return None, None, None, grad_w, grad_u, grad_k, grad_v


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, n_layer=None, layer_id=None, init_mode='fancy', key_norm=False,
                 scan_schemes=None, recurrence=2):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        attn_sz = n_embd
        self.device = None
        self.recurrence = int(recurrence)
        self.scan_schemes = scan_schemes or [('top-left', 'horizontal'), ('bottom-right', 'vertical')]
        self.dwconv = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)
        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(n_embd)
        else:
            self.key_norm = None
        self.output = nn.Linear(attn_sz, n_embd, bias=False)
        self.spatial_decay = nn.Parameter(torch.zeros((self.recurrence, self.n_embd)))
        self.spatial_first = nn.Parameter(torch.zeros((self.recurrence, self.n_embd)))

    def run_wkv(self, B, T, C, w, u, k, v):
        dtype = k.dtype
        device = k.device
        eps = 1e-6
        w = torch.sigmoid(w).to(dtype=dtype, device=device).view(1, C)
        w = torch.clamp(w, 0.0 + 1e-5, 1.0 - 1e-5)
        u = torch.tanh(u).to(dtype=dtype, device=device).view(1, C)

        y = torch.zeros((B, T, C), device=device, dtype=dtype)
        num = torch.zeros((B, C), device=device, dtype=dtype)
        den = torch.zeros((B, C), device=device, dtype=dtype)
        state = torch.zeros((B, C), device=device, dtype=dtype)
        for t in range(T):
            kt = k[:, t, :]
            vt = v[:, t, :]
            num = num * w + kt * vt
            den = den * w + torch.abs(kt)
            state = u * state + num / (den + eps)
            state = torch.clamp(state, -1e4, 1e4)
            y[:, t, :] = state
        y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
        return y

    def get_zigzag_indices(self, h, w, start='top-left', direction='horizontal'):
        indices = []
        if start == 'top-left':
            row_start = 0
            col_start = 0
            row_step = 1
            col_step = 1 if direction == 'horizontal' else 1
        elif start == 'top-right':
            row_start = 0
            col_start = w - 1
            row_step = 1
            col_step = -1 if direction == 'horizontal' else -1
        elif start == 'bottom-left':
            row_start = h - 1
            col_start = 0
            row_step = -1
            col_step = 1 if direction == 'horizontal' else 1
        elif start == 'bottom-right':
            row_start = h - 1
            col_start = w - 1
            row_step = -1
            col_step = -1 if direction == 'horizontal' else -1

        for i in range(h):
            current_row = row_start + row_step * i
            if direction == 'horizontal':
                if current_row % 2 == 0:
                    cols = list(range(w))
                else:
                    cols = list(range(w - 1, -1, -1))
                for col in cols:
                    indices.append(current_row * w + col)
            elif direction == 'vertical':
                if (col_start + col_step * i) % 2 == 0:
                    rows = list(range(h))
                else:
                    rows = list(range(h - 1, -1, -1))
                for row in rows:
                    indices.append(row * w + (col_start + col_step * i))
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def jit_func(self, x, resolution, scan_scheme):
        h, w = resolution
        start, direction = scan_scheme
        zigzag_order = self.get_zigzag_indices(h, w, start=start, direction=direction)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)

        x = rearrange(x, 'b c h w -> b c (h w)')
        x = x[..., zigzag_order]
        x = rearrange(x, 'b c (h w) -> b (h w) c', h=h, w=w)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        selected_scheme = self.scan_schemes[self.layer_id % len(self.scan_schemes)]
        sr, k, v = self.jit_func(x, resolution, selected_scheme)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = self.run_wkv(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                h, w = resolution
                new_h, new_w = (h, w) if selected_scheme[1] == 'horizontal' else (w, h)
                zigzag_order = self.get_zigzag_indices(new_h, new_w, start=selected_scheme[0],
                                                      direction=selected_scheme[1])
                k = rearrange(k, 'b (h w) c -> b c h w', h=h, w=w)
                k = rearrange(k, 'b c h w -> b c (h w)')[..., zigzag_order]
                k = rearrange(k, 'b c (h w) -> b (h w) c', h=new_h, w=new_w)

                v = rearrange(v, 'b (h w) c -> b c h w', h=h, w=w)
                v = rearrange(v, 'b c h w -> b c (h w)')[..., zigzag_order]
                v = rearrange(v, 'b c (h w) -> b (h w) c', h=new_h, w=new_w)

                v = self.run_wkv(B, T, C, self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                k = rearrange(k, 'b (h w) c -> b (h w) c', h=h, w=w)
                v = rearrange(v, 'b (h w) c -> b (h w) c', h=h, w=w)

        x = v
        if self.key_norm is not None:
            x = self.key_norm(x)
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, n_layer=None, layer_id=None, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        if key_norm:
            self.key_norm = nn.LayerNorm(hidden_sz)
        else:
            self.key_norm = None
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = q_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        k = self.key(x)
        k = torch.square(torch.relu(k))
        if self.key_norm is not None:
            k = self.key_norm(k)
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x

