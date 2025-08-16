import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math
from .vision_transformer import CrossBlock, Mlp, ParallelCrossBlock
from torch.nn.init import trunc_normal_
from utils import info_nce


class LayerNorm2D(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConditionHead(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=512):
        super().__init__()
        self.linear1 = nn.Linear(input_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(input_dim * 2, hidden_dim)
        self.act = nn.GELU()
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, a, b, cls):
        a = self.linear1(torch.cat([a, cls.expand(-1, a.shape[1], -1)], dim=-1))
        b = self.linear2(torch.cat([b, cls.expand(-1, a.shape[1], -1)], dim=-1))

        out = self.act(a) * self.act(b)
        out = self.out(out)
        return out


class FADE(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            interested_layers=[5, 8, 11],
            decoder_embed_dim=32,
            remove_class_token=False,
            num_anomaly_registers=0,
    ) -> None:
        super(FADE, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck

        self.fuser = nn.Linear(len(interested_layers) * self.encoder.embed_dim, self.encoder.embed_dim)

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(self.encoder.embed_dim, self.encoder.patch_size ** 2 * self.decoder_embed_dim,
                                       bias=True)  # decoder to patch
        self.decoder_pred = nn.Sequential(
            nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1),
            LayerNorm2D(self.decoder_embed_dim),
            nn.GELU(),
            nn.Conv2d(self.decoder_embed_dim, 1, kernel_size=1, bias=True),  # decoder to patch
        )
        self.anomaly_registers = nn.Parameter(
            torch.zeros(1, num_anomaly_registers, self.encoder.embed_dim)) if num_anomaly_registers else None

        self.interested_layers = interested_layers
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        self.init_weights()
        self.x_support = None

    def init_weights(self):
        if self.anomaly_registers is not None:
            trunc_normal_(self.anomaly_registers, std=0.02)

        trainable = nn.ModuleList(
            [self.bottleneck, self.decoder_pred, self.decoder_embed, self.fuser])

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        trunc_normal_(self.decoder_pred[-1].weight, std=1e-4)
        nn.init.constant_(self.decoder_pred[-1].bias, -2)

    def forward(self, queries, supports, memorize_supports=False, use_memory=False):
        """Query Images
        queries: Nx3xHxW
        """
        with torch.no_grad():
            x = self.encoder.prepare_tokens(queries)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.interested_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.interested_layers:
                    en_list.append(x)
            side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))
            if self.remove_class_token:
                en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x_query = self.fuser(torch.cat(en_list, dim=-1))
        # x_query = self.fuse_feature(en_list)

        """Support Images
        supports: NxSx3xHxW
        """
        if self.training or memorize_supports or not use_memory:
            with torch.no_grad():
                N, S, _, H, W = supports.shape
                supports = supports.view(N * S, 3, H, W)
                x = self.encoder.prepare_tokens(supports)
                en_list = []
                for i, blk in enumerate(self.encoder.blocks):
                    if i <= self.interested_layers[-1]:
                        x = blk(x)
                    else:
                        continue
                    if i in self.interested_layers:
                        en_list.append(x)
                if self.remove_class_token:
                    en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

            x_support = self.fuser(torch.cat(en_list, dim=-1))
            # x_support = self.fuse_feature(en_list)

            x_support = x_support.reshape(N, -1, x_support.shape[-1])

            if self.anomaly_registers is not None:
                x_support = torch.cat([self.anomaly_registers.expand(x_support.shape[0], -1, -1), x_support], dim=1)

            if self.training and x_support.shape[0] < x_query.shape[0]:
                x_support = torch.cat([x_support, x_support], dim=0)

            if not self.training and memorize_supports:
                self.x_support = x_support

        if not self.training and use_memory:
            x_support = self.x_support

        if not self.training and x_support.shape[0] == 1:
            x_support = x_support.repeat(x_query.shape[0], 1, 1)

        """Cross Attentions"""
        x_query_original = x_query
        for i, blk in enumerate(self.bottleneck):
            if isinstance(blk, (CrossBlock)):
                x_query = blk(x_query, x_support)
            else:
                x_query = blk(x_query)

        x = F.normalize(x_query_original, dim=-1) * F.normalize(x_query, dim=-1)

        if not self.remove_class_token:  # class tokens have not been removed above
            x = x[:, 1 + self.encoder.num_register_tokens:, :]

        p = self.encoder.patch_size
        x = self.decoder_embed(x)  # BxLxC*p*p
        x = x.reshape(shape=(x.shape[0], side, side, p, p, self.decoder_embed_dim))  # B,h,w,p,p,C
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(shape=(x.shape[0], -1, side * p, side * p))
        x = self.decoder_pred(x)
        x = F.sigmoid(x)
        return x

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class FADEv2(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            interested_layers=[5, 8, 11],
            decoder_embed_dim=32,
            remove_class_token=False,
            num_anomaly_registers=4,
    ) -> None:
        super(FADEv2, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder

        self.fuser = nn.Linear(len(interested_layers) * self.encoder.embed_dim, self.encoder.embed_dim)

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(self.encoder.embed_dim, self.encoder.patch_size ** 2 * self.decoder_embed_dim,
                                       bias=True)  # decoder to patch
        self.decoder_pred = nn.Sequential(
            nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1),
            LayerNorm2D(self.decoder_embed_dim),
            nn.GELU(),
            nn.Conv2d(self.decoder_embed_dim, 1, kernel_size=1, bias=True),  # decoder to patch
        )
        self.anomaly_registers = nn.Parameter(
            torch.zeros(1, num_anomaly_registers, self.encoder.embed_dim)) if num_anomaly_registers else None

        self.interested_layers = interested_layers
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        self.init_weights()
        self.x_support = None

    def init_weights(self):
        if self.anomaly_registers is not None:
            trunc_normal_(self.anomaly_registers, std=0.02)

        trainable = nn.ModuleList(
            [self.bottleneck, self.decoder, self.decoder_pred, self.decoder_embed, self.fuser])

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        trunc_normal_(self.decoder_pred[-1].weight, std=1e-4)
        nn.init.constant_(self.decoder_pred[-1].bias, -2)

    def forward(self, queries, supports, memorize_supports=False, use_memory=False):
        """Query Images
        queries: Nx3xHxW
        """
        with torch.no_grad():
            x = self.encoder.prepare_tokens(queries)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.interested_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.interested_layers:
                    en_list.append(x)
            side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))
            if self.remove_class_token:
                en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x_query = self.fuser(torch.cat(en_list, dim=-1))
        # x_query = self.fuse_feature(en_list)

        """Support Images
        supports: NxSx3xHxW
        """
        if self.training or memorize_supports or not use_memory:
            with torch.no_grad():
                N, S, _, H, W = supports.shape
                supports = supports.view(N * S, 3, H, W)
                x = self.encoder.prepare_tokens(supports)
                en_list = []
                for i, blk in enumerate(self.encoder.blocks):
                    if i <= self.interested_layers[-1]:
                        x = blk(x)
                    else:
                        continue
                    if i in self.interested_layers:
                        en_list.append(x)
                if self.remove_class_token:
                    en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

            x_support = self.fuser(torch.cat(en_list, dim=-1))
            # x_support = self.fuse_feature(en_list)

            x_support_list = []
            for i, blk in enumerate(self.bottleneck):
                x_support = blk(x_support)
                x_support_reshape = x_support.reshape(N, -1, x_support.shape[-1])
                x_support_list.append(x_support_reshape)
            x_support_list = x_support_list[::-1]

            if self.training and x_support_list[0].shape[0] < x_query.shape[0]:
                x_support_list = [torch.cat([x_support, x_support], dim=0) for x_support in x_support_list]

            if not self.training and memorize_supports:
                self.x_support_list = x_support_list

        if not self.training and use_memory:
            x_support_list = self.x_support_list

        if not self.training and x_support_list[0].shape[0] == 1:
            x_support_list = [x_support.repeat(x_query.shape[0], 1, 1) for x_support in x_support_list]

        """Self Attentions"""
        x_query_list = []
        for i, blk in enumerate(self.bottleneck):
            x_query = blk(x_query)
            x_query_list.append(x_query)
        x_query_list = x_query_list[::-1]

        """Cross Attentions"""
        for i, blk in enumerate(self.decoder):
            x_query = blk(x_query_list[i], x_support_list[i], source=x_query)

        if not self.remove_class_token:  # class tokens have not been removed above
            x_query = x_query[:, 1 + self.encoder.num_register_tokens:, :]

        p = self.encoder.patch_size
        x = self.decoder_embed(x_query)  # BxLxC*p*p
        x = x.reshape(shape=(x.shape[0], side, side, p, p, self.decoder_embed_dim))  # B,h,w,p,p,C
        x = torch.einsum('bhwpqc->bchpwq', x)
        x = x.reshape(shape=(x.shape[0], -1, side * p, side * p))
        x = self.decoder_pred(x)
        x = F.sigmoid(x)
        return x

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class FADEv3(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            head,
            interested_layers=[5, 8, 11],
            remove_class_token=False,
            num_anomaly_registers=0,
    ) -> None:
        super(FADEv3, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.head = head

        # self.decoder_pred = nn.Sequential(
        #     # nn.ConvTranspose2d(self.encoder.embed_dim, self.decoder_embed_dim,
        #     #                    kernel_size=self.encoder.patch_size // 2, stride=self.encoder.patch_size // 2),
        #     nn.Conv2d(self.decoder_embed_dim, self.decoder_embed_dim, kernel_size=3, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(self.decoder_embed_dim, 1, kernel_size=1, bias=True),  # decoder to patch
        # )

        self.anomaly_registers = nn.Parameter(
            torch.zeros(1, num_anomaly_registers, self.encoder.embed_dim)) if num_anomaly_registers else None

        self.interested_layers = interested_layers
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        self.init_weights()
        self.x_support = None

    def init_weights(self):
        trainable = nn.ModuleList(
            [self.bottleneck, self.decoder, self.head])

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # nn.init.constant_(self.head.weight, -1.)
        # nn.init.constant_(self.head.bias, -1.)
        nn.init.constant_(self.head.weight, 2.)
        nn.init.constant_(self.head.bias, -1.)

    def forward(self, queries, supports, memorize_supports=False, use_memory=False):
        """Query Images
        queries: Nx3xHxW
        """
        with torch.no_grad():
            x = self.encoder.prepare_tokens(queries)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.interested_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.interested_layers:
                    en_list.append(x)
            side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))
            if self.remove_class_token:
                en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x_query_original = self.fuse_feature(en_list)
        x_query = self.bottleneck(x_query_original)

        """Support Images
        supports: NxSx3xHxW
        """
        if self.training or memorize_supports or not use_memory:
            with torch.no_grad():
                N, S, _, H, W = supports.shape
                supports = supports.view(N * S, 3, H, W)
                x = self.encoder.prepare_tokens(supports)
                en_list = []
                for i, blk in enumerate(self.encoder.blocks):
                    if i <= self.interested_layers[-1]:
                        x = blk(x)
                    else:
                        continue
                    if i in self.interested_layers:
                        en_list.append(x)
                if self.remove_class_token:
                    en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

            x_support_original = self.fuse_feature(en_list)
            x_support = self.bottleneck(x_support_original)

            x_support = x_support.reshape(N, -1, x_support.shape[-1])
            x_support_original = x_support_original.reshape(N, -1, x_support_original.shape[-1])

            if self.anomaly_registers is not None:
                x_support = torch.cat([self.anomaly_registers.expand(x_support.shape[0], -1, -1), x_support], dim=1)

            if self.training and x_support.shape[0] < x_query.shape[0]:
                x_support = torch.cat([x_support, x_support], dim=0)
                x_support_original = torch.cat([x_support_original, x_support_original], dim=0)

            if not self.training and memorize_supports:
                self.x_support = x_support
                self.x_support_original = x_support_original

        if not self.training and use_memory:
            x_support = self.x_support
            x_support_original = self.x_support_original

        if not self.training and x_support.shape[0] == 1:
            x_support = x_support.repeat(x_query.shape[0], 1, 1)
            x_support_original = x_support_original.repeat(x_query.shape[0], 1, 1)

        """Cross Attentions"""
        de_list = []
        for i, blk in enumerate(self.decoder):
            if isinstance(blk, (CrossBlock)):
                x_query = blk(x_query, x_support)
            else:
                x_query = blk(x_query)
            de_list.append(x_query)
        # x_query = self.fuse_feature(de_list)

        if not self.remove_class_token:  # class tokens have not been removed above
            x_query = x_query[:, 1 + self.encoder.num_register_tokens:, :]
            x_query_original = x_query_original[:, 1 + self.encoder.num_register_tokens:, :]

        x_query_original = F.normalize(x_query_original, dim=-1)
        x_query = F.normalize(x_query, dim=-1)
        x_support_original = F.normalize(x_support_original, dim=-1)

        cross_distance = 1 - (x_query_original @ x_support_original.transpose(-2, -1))
        cross_distance = torch.min(cross_distance, dim=-1)[0]

        x_query_original = x_query_original.permute(0, 2, 1).reshape([queries.shape[0], -1, side, side]).contiguous()
        x_query = x_query.permute(0, 2, 1).reshape([queries.shape[0], -1, side, side]).contiguous()

        # x = x_query_original * x_query

        recon_distance = 1 - torch.cosine_similarity(x_query, x_query_original, dim=1).unsqueeze(1)
        cross_distance = cross_distance.reshape([queries.shape[0], 1, side, side]).contiguous()

        x = self.head(recon_distance * cross_distance)
        x = F.sigmoid(x)

        return x, recon_distance * cross_distance

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class FADEv4(nn.Module):
    def __init__(
            self,
            encoder,
            head,
            interested_layers=[5, 8, 11],
            remove_class_token=False,
            rotate_supports=False,
    ) -> None:
        super(FADEv4, self).__init__()
        self.encoder = encoder
        self.head = head
        self.interested_layers = interested_layers
        self.remove_class_token = remove_class_token
        self.rotate_supports = rotate_supports
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        self.x_support = None
        self.x_support_cls = None

    def forward(self, queries, supports, memorize_supports=False, use_memory=False):
        """Query Images
        queries: Nx3xHxW
        """
        with torch.no_grad():
            x = self.encoder.prepare_tokens(queries)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.interested_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.interested_layers:
                    en_list.append(x)

            side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        x_query = self.fuse_feature(en_list)
        x_query = x_query[:, 1 + self.encoder.num_register_tokens:, :]
        # x_query = self.neighbour_pool(x_query, side)
        """Support Images
        supports: NxSx3xHxW
        """
        if self.rotate_supports:
            supports_list = [supports]
            for i in range(3):
                supports_list.append(torch.rot90(supports, k=1, dims=[-2, -1]))
            supports = torch.cat(supports_list, dim=1)

        if self.training or memorize_supports or not use_memory:
            with torch.no_grad():
                N, S, _, H, W = supports.shape
                supports = supports.view(N * S, 3, H, W)
                x = self.encoder.prepare_tokens(supports)
                en_list = []
                for i, blk in enumerate(self.encoder.blocks):
                    if i <= self.interested_layers[-1]:
                        x = blk(x)
                    else:
                        continue
                    if i in self.interested_layers:
                        en_list.append(x)

            x_support = self.fuse_feature(en_list)

            x_support_cls = x_support[:, :1, :]
            x_support = x_support[:, 1 + self.encoder.num_register_tokens:, :]
            # x_support = self.neighbour_pool(x_support, side)

            x_support = x_support.reshape(N, -1, x_support.shape[-1])
            x_support_cls = x_support_cls.reshape(N, -1, x_support.shape[-1])

            if self.training and x_support.shape[0] < x_query.shape[0]:
                x_support = torch.cat([x_support, x_support], dim=0)
                x_support_cls = torch.cat([x_support_cls, x_support_cls], dim=0)

            if not self.training and memorize_supports:
                self.x_support = x_support
                self.x_support_cls = x_support_cls

        if not self.training and use_memory:
            x_support = self.x_support
            x_support_cls = self.x_support_cls

        if not self.training and x_support.shape[0] == 1:
            x_support = x_support.repeat(x_query.shape[0], 1, 1)
            x_support_cls = x_support_cls.repeat(x_query.shape[0], 1, 1)

        x_support_cls = torch.mean(x_support_cls, dim=1, keepdim=True)
        x_support_cls = F.normalize(x_support_cls, dim=-1)

        # x_support_cls = F.normalize(x_support_cls, dim=-1)
        # x_query = x_query - x_support_cls
        # x_support = x_support - x_support_cls

        x_query = F.normalize(x_query, dim=-1)
        x_support = F.normalize(x_support, dim=-1)

        cross_distance = 1 - (x_query @ x_support.transpose(-2, -1))
        cross_distance, min_idx = torch.min(cross_distance, dim=-1)
        nearest_support = x_support[torch.arange(x_query.shape[0]).unsqueeze(1), min_idx]

        x = torch.cat([x_query, nearest_support, x_support_cls.expand(-1, x_query.shape[1], -1)], dim=-1)
        x = self.head(x)
        pred = torch.sigmoid(x)

        cross_distance = cross_distance.reshape([queries.shape[0], 1, side, side])
        pred = pred.reshape([queries.shape[0], 1, side, side])

        return pred * cross_distance, pred

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class RADE(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            bridge,
            decoder,
            interested_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            remove_class_token=False,
            num_anomaly_registers=0,
    ) -> None:
        super(RADE, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.bridge = bridge
        self.decoder = decoder

        self.anomaly_registers = nn.Parameter(
            torch.zeros(1, num_anomaly_registers, self.encoder.embed_dim)) if num_anomaly_registers else None

        self.interested_layers = interested_layers
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        self.init_weights()
        self.x_support = None

    def init_weights(self):
        trainable = nn.ModuleList([self.bottleneck, self.bridge, self.decoder])

        for m in trainable.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, queries, supports, memorize_supports=False, use_memory=False):
        """Query Images
        queries: Nx3xHxW
        """
        with torch.no_grad():
            x = self.encoder.prepare_tokens(queries)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.interested_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.interested_layers:
                    en_list.append(x)
            side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))
            if self.remove_class_token:
                en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x_query = self.fuse_feature(en_list)
        x_query_original = x_query.detach()

        """Noisy Bottlenecks"""
        x_query = self.bottleneck(x_query)

        """Support Images
        supports: NxSx3xHxW
        """
        if self.training or memorize_supports or not use_memory:
            with torch.no_grad():
                N, S, _, H, W = supports.shape
                supports = supports.view(N * S, 3, H, W)
                x = self.encoder.prepare_tokens(supports)
                en_list = []
                for i, blk in enumerate(self.encoder.blocks):
                    if i <= self.interested_layers[-1]:
                        x = blk(x)
                    else:
                        continue
                    if i in self.interested_layers:
                        en_list.append(x)
                if self.remove_class_token:
                    en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

            x_support = self.fuse_feature(en_list)
            x_support = x_support.reshape(N, -1, x_support.shape[-1])

            x_support = self.bridge(x_support)

            if self.anomaly_registers is not None:
                x_support = torch.cat([self.anomaly_registers.expand(x_support.shape[0], -1, -1), x_support], dim=1)

            if self.training and x_support.shape[0] < x_query.shape[0]:
                x_support = torch.cat([x_support, x_support], dim=0)

            if not self.training and memorize_supports:
                self.x_support = x_support

        if not self.training and use_memory:
            x_support = self.x_support

        if not self.training and x_support.shape[0] == 1:
            x_support = x_support.repeat(x_query.shape[0], 1, 1)

        """Cross Attentions"""
        de_list = []
        for i, blk in enumerate(self.decoder):
            if isinstance(blk, (CrossBlock, ParallelCrossBlock)):
                x_query = blk(x_query, x_support)
            else:
                x_query = blk(x_query)
            de_list.append(x_query)
        # x_query = self.fuse_feature(de_list)
        x_query = de_list[-1]

        if not self.remove_class_token:  # class tokens have not been removed above
            x_query = x_query[:, 1 + self.encoder.num_register_tokens:, :]
            x_query_original = x_query_original[:, 1 + self.encoder.num_register_tokens:, :]

        x_query_original = x_query_original.permute(0, 2, 1).reshape([queries.shape[0], -1, side, side]).contiguous()
        x_query = x_query.permute(0, 2, 1).reshape([queries.shape[0], -1, side, side]).contiguous()

        return [x_query_original], [x_query]

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class DADE(nn.Module):
    def __init__(
            self,
            encoder,
            interested_layers=[5, 8, 11],
            remove_class_token=False,
            rotate_supports=False,
            use_vv_attn=False,
    ) -> None:
        super(DADE, self).__init__()
        self.encoder = encoder

        self.interested_layers = interested_layers
        self.remove_class_token = remove_class_token
        self.rotate_supports = rotate_supports
        self.use_vv_attn = use_vv_attn
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        self.x_support = None
        self.x_support_cls = None

    def forward(self, queries, supports, memorize_supports=False, use_memory=False, aug=False):
        """Query Images
        queries: Nx3xHxW
        """
        with torch.no_grad():
            x = self.encoder.prepare_tokens(queries)
            en_list = []
            for i, blk in enumerate(self.encoder.blocks):
                if i <= self.interested_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.interested_layers:
                    if x.size(0) != queries.size(0):
                        en_list.append(x.permute(1, 0, 2))
                    else:
                        en_list.append(x)

            side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        x_query = self.fuse_feature(en_list)
        x_query = x_query[:, 1 + self.encoder.num_register_tokens:, :]
        # x_query = self.neighbour_pool(x_query, side)
        """Support Images
        supports: NxSx3xHxW
        """
        if self.rotate_supports:
            supports_list = [supports]
            for i in range(3):
                supports_list.append(torch.rot90(supports, k=i+1, dims=[-2, -1]))
                
            supports_list.append(torch.flip(supports, dims=[-2])) # 垂直翻转
            supports_list.append(torch.flip(supports, dims=[-1])) # 水平翻转
            supports = torch.cat(supports_list, dim=1)

        # if self.training or memorize_supports or not use_memory:
        if memorize_supports or not use_memory:
            with torch.no_grad():
                N, S, _, H, W = supports.shape
                supports = supports.view(N * S, 3, H, W)
                x = self.encoder.prepare_tokens(supports)
                en_list = []
                for i, blk in enumerate(self.encoder.blocks):
                    if i <= self.interested_layers[-1]:
                        x = blk(x)
                    else:
                        continue
                    if i in self.interested_layers:
                        en_list.append(x)

            x_support = self.fuse_feature(en_list)
            x_support_cls = x_support[:, :1, :]
            x_support = x_support[:, 1 + self.encoder.num_register_tokens:, :]
            # x_support = self.neighbour_pool(x_support, side)

            x_support = x_support.reshape(N, -1, x_support.shape[-1])
            x_support_cls = x_support_cls.reshape(N, -1, x_support.shape[-1])

            # if not self.training and memorize_supports:
            if memorize_supports:
                if aug==1:
                    self.x_support_aug1 = x_support
                    self.x_support_cls_aug1 = x_support_cls
                elif aug==2:
                    self.x_support_aug2 = x_support
                    self.x_support_cls_aug2 = x_support_cls
                elif aug==3:
                    self.x_support_aug3 = x_support
                    self.x_support_cls_aug3 = x_support_cls                    
                else:
                    self.x_support = x_support
                    self.x_support_cls = x_support_cls

        # if not self.training and use_memory:
        if use_memory:
            if aug==1:
                x_support = self.x_support_aug1
                x_support_cls = self.x_support_cls_aug1
            elif aug==2:
                x_support = self.x_support_aug2
                x_support_cls = self.x_support_cls_aug2
            elif aug==3:
                x_support = self.x_support_aug3
                x_support_cls = self.x_support_cls_aug3
            else:
                x_support = self.x_support
                x_support_cls = self.x_support_cls

        # if not self.training and x_support.shape[0] == 1:
        if x_support.shape[0] == 1:
            x_support = x_support.repeat(x_query.shape[0], 1, 1)
            x_support_cls = x_support_cls.repeat(x_query.shape[0], 1, 1)

        x_support_cls = torch.mean(x_support_cls, dim=1, keepdim=True)

        x_query = F.normalize(x_query, dim=-1)
        x_support = F.normalize(x_support, dim=-1)

        cross_distance = 1 - (x_query @ x_support.transpose(-2, -1))
        cross_distance, min_idx = torch.min(cross_distance, dim=-1)

        # nearest_support = x_support[torch.arange(x_query.shape[0]).unsqueeze(1), min_idx]

        cross_distance = cross_distance.reshape([queries.shape[0], 1, side, side])

        return cross_distance

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class DADEv2(nn.Module):
    def __init__(
            self,
            encoder,
            interested_layers=[5, 8, 11],
            remove_class_token=False,
            rotate_supports=False,
            use_vv_attn=False,
    ) -> None:
        super(DADEv2, self).__init__()
        self.encoder = encoder

        self.interested_layers = interested_layers
        self.remove_class_token = remove_class_token
        self.rotate_supports = rotate_supports
        self.use_vv_attn = use_vv_attn
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

        self.x_support = None
        self.x_support_cls = None

    def forward(self, queries, supports, memorize_supports=False, use_memory=False):
        """Query Images
        queries: Nx3xHxW
        """

        x = self.encoder.prepare_tokens(queries)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.interested_layers[-1]:
                x = blk(x)
            else:
                continue
            if i in self.interested_layers:
                en_list.append(x)

        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        x_query = self.fuse_feature(en_list)
        x_query = x_query[:, 1 + self.encoder.num_register_tokens:, :]
        # x_query = self.neighbour_pool(x_query, side)
        """Support Images
        supports: NxSx3xHxW
        """
        if self.rotate_supports:
            supports_list = [supports]
            for i in range(3):
                supports_list.append(torch.rot90(supports, k=1, dims=[-2, -1]))
            supports = torch.cat(supports_list, dim=1)

        if self.training or memorize_supports or not use_memory:
            with torch.no_grad():
                N, S, _, H, W = supports.shape
                supports = supports.view(N * S, 3, H, W)
                x = self.encoder.prepare_tokens(supports)
                en_list = []
                for i, blk in enumerate(self.encoder.blocks):
                    if i <= self.interested_layers[-1]:
                        x = blk(x)
                    else:
                        continue
                    if i in self.interested_layers:
                        en_list.append(x)

            x_support = self.fuse_feature(en_list)
            x_support_cls = x_support[:, :1, :]
            x_support = x_support[:, 1 + self.encoder.num_register_tokens:, :]
            # x_support = self.neighbour_pool(x_support, side)

            x_support = x_support.reshape(N, -1, x_support.shape[-1])
            x_support_cls = x_support_cls.reshape(N, -1, x_support.shape[-1])

            # if self.training and x_support.shape[0] < x_query.shape[0]:
            #     x_support = torch.cat([x_support, x_support], dim=0)

            if not self.training and memorize_supports:
                self.x_support = x_support
                self.x_support_cls = x_support_cls

        if not self.training and use_memory:
            x_support = self.x_support
            x_support_cls = self.x_support_cls

        if not self.training and x_support.shape[0] == 1:
            x_support = x_support.repeat(x_query.shape[0], 1, 1)
            x_support_cls = x_support_cls.repeat(x_query.shape[0], 1, 1)

        x_support_cls = torch.mean(x_support_cls, dim=1, keepdim=True)

        # x_support_cls = F.normalize(x_support_cls, dim=-1)
        # x_query = x_query - x_support_cls
        # x_support = x_support - x_support_cls

        x_query = F.normalize(x_query, dim=-1)
        x_support = F.normalize(x_support, dim=-1)

        cross_distance = 1 - (x_query @ x_support.transpose(-2, -1))
        cross_distance, min_idx = torch.min(cross_distance, dim=-1)

        nearest_support = x_support[torch.arange(x_query.shape[0]).unsqueeze(1), min_idx]

        dim = x_query.shape[-1]
        loss = info_nce(x_query.reshape(-1, dim), nearest_support.reshape(-1, dim), x_support.reshape(-1, dim),
                        temperature=0.1, negative_mode='unpaired')

        cross_distance = cross_distance.reshape([queries.shape[0], 1, side, side])
        # x_query = x_query.permute(0, 2, 1).reshape([queries.shape[0], -1, side, side]).contiguous()
        # nearest_support = nearest_support.permute(0, 2, 1).reshape([queries.shape[0], -1, side, side]).contiguous()

        return cross_distance, loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


def update_moving_average(ma_model, current_model, momentum=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight)

    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)


def update_average(old, new, momentum=0.99):
    if old is None:
        return new
    return old * momentum + (1 - momentum) * new


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
