"""
Added get selfattention from all layer

Mostly copy-paster from DINO (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
and timm library (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)

"""
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super(BatchNorm1d, self).forward(x)
        x = x.permute(0, 2, 1)
        return x


class ShuffleDrop(nn.Module):
    def __init__(self, p=0.):
        super(ShuffleDrop, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            N, P, C = x.shape
            idx = torch.randperm(N * P)
            shuffle_x = x.reshape(-1, C)[idx, :].view(x.size()).detach()
            drop_mask = torch.bernoulli(torch.ones_like(x) * self.p).bool()
            x[drop_mask] = shuffle_x[drop_mask]
        return x


class MeanDrop(nn.Module):
    def __init__(self, p=0.):
        super(MeanDrop, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mean = x.mean()
            drop_mask = torch.bernoulli(torch.ones_like(x) * self.p).bool()
            x[drop_mask] = mean
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DropKey(nn.Module):
    """DropKey
    """

    def __init__(self, p=0.):
        super(DropKey, self).__init__()
        self.p = p

    def forward(self, attn):
        if self.training and self.p > 0.:
            m_r = torch.ones_like(attn) * self.p
            attn = attn + torch.bernoulli(m_r) * -1e12
        return attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class bMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 grad=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.grad = grad

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # x = self.grad * x + (1 - self.grad) * x.detach()
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = DropKey(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn = self.attn_drop(attn)
        attn = attn.softmax(dim=-1)

        if attn_mask is not None:
            attn = attn.clone()
            attn[:, :, attn_mask == 0.] = 0.

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = nn.functional.elu(q) + 1.
        k = nn.functional.elu(k) + 1.

        attn = (q @ k.transpose(-2, -1))
        attn = self.attn_drop(attn)

        if attn_mask is not None:
            attn[:, :, attn_mask == 0.] = 0.

        attn = attn / (torch.sum(attn, dim=-1, keepdim=True))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class LinearAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = nn.functional.elu(q) + 1.
        k = nn.functional.elu(k) + 1.

        kv = torch.einsum('...sd,...se->...de', k, v)
        z = 1.0 / torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))
        x = torch.einsum('...de,...sd,...s->...se', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn=Attention, init_values=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

    def forward(self, x, attn_mask=None):
        y = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ls1(y)
        x = x + self.ls2(self.mlp(self.norm2(x)))

        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 cosine_similarity=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = DropKey(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv_bias = qkv_bias

        self.cosine_similarity = cosine_similarity

    def forward(self, x, support, attn_mask=None):
        B, N, C = x.shape

        if self.qkv_bias:
            q = F.linear(x, self.qkv.weight[:C], self.qkv.bias[:C])
            k = F.linear(support, self.qkv.weight[C:2 * C], self.qkv.bias[C:2 * C])
            v = F.linear(support, self.qkv.weight[2 * C:], self.qkv.bias[2 * C:])
        else:
            q = F.linear(x, self.qkv.weight[:C], None)
            k = F.linear(support, self.qkv.weight[C:2 * C], None)
            v = F.linear(support, self.qkv.weight[2 * C:], None)

        if self.cosine_similarity:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1)) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn = self.attn_drop(attn)
        attn = attn.softmax(dim=-1)
        # attn = softmax_1(attn)

        if attn_mask is not None:
            attn = attn.clone()
            attn[:, :, attn_mask == 0.] = 0.

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VVCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 cosine_similarity=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = DropKey(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.qkv_bias = qkv_bias

        self.cosine_similarity = cosine_similarity

    def forward(self, x, support, attn_mask=None):
        B, N, C = x.shape

        q = self.v_linear(x)
        k = self.v_linear(support)
        v = k

        if self.cosine_similarity:
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
            attn = (q @ k.transpose(-2, -1)) * self.scale
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn = self.attn_drop(attn)
        attn = attn.softmax(dim=-1)
        # attn = softmax_1(attn)

        if attn_mask is not None:
            attn = attn.clone()
            attn[:, :, attn_mask == 0.] = 0.

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def softmax_1(x, dim=-1):
    assert dim == -1 or dim == -2
    if dim == -1:
        x = F.pad(x, (0, 1), value=0)
        x = x.softmax(dim=-1)
        x = x[..., :-1]
    else:
        x = F.pad(x, (0, 0, 0, 1), value=0)
        x = x.softmax(dim=-2)
        x = x[..., :-1, :]
    return x


class CrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn=CrossAttention, support_norm=True,
                 init_values=None, cosine_similarity=False, no_residual=False):
        super().__init__()
        self.no_residual = no_residual
        self.norm1 = norm_layer(dim)
        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         attn_drop=attn_drop, proj_drop=drop, cosine_similarity=cosine_similarity)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        if support_norm == True:
            self.support_norm = norm_layer(dim)
        else:
            self.support_norm = None

    def forward(self, query, support, source=None, attn_mask=None):
        if source is not None:
            x = source
        else:
            x = query
        if self.support_norm is not None:
            y = self.attn(self.norm1(query), self.support_norm(support), attn_mask=attn_mask)
        else:
            y = self.attn(self.norm1(query), self.norm1(support), attn_mask=attn_mask)
        if self.no_residual:
            x = self.ls1(y)
        else:
            x = x + self.ls1(y)
        x = x + self.ls2(self.mlp(self.norm2(x)))

        return x


class ParallelCrossBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn=CrossAttention, support_norm=True,
                 init_values=None, cosine_similarity=False, no_residual=False):
        super().__init__()
        self.no_residual = no_residual
        self.norm1 = norm_layer(dim)
        self.attn = attn(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         attn_drop=attn_drop, proj_drop=drop, cosine_similarity=cosine_similarity)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        if support_norm == True:
            self.support_norm = norm_layer(dim)
        else:
            self.support_norm = None

    def forward(self, query, support, source=None, attn_mask=None):
        if source is not None:
            x = source
        else:
            x = query

        if self.support_norm is not None:
            y = self.attn(self.norm1(query), self.support_norm(support), attn_mask=attn_mask)
        else:
            y = self.attn(self.norm1(query), self.norm1(support), attn_mask=attn_mask)

        y = self.ls2(self.mlp(self.norm2(x))) + self.ls1(y)

        if self.no_residual:
            x = y
        else:
            x = x + y

        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values=1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
