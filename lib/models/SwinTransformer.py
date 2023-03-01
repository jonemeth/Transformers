import math
from typing import Tuple, List

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.common import TransformerEncoder, RelativePositionEncoding


# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# https://github.com/berniwal/swin-transformer-pytorch/blob/master/swin_transformer_pytorch/swin_transformer.py
# https://juliusruseckas.github.io/ml/swin-cifar10.html


def create_upper_lower_mask(window_size, displacement):
    mask = torch.zeros(window_size ** 2, window_size ** 2)
    mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
    mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')
    return mask


def create_left_right_mask(window_size, displacement):
    mask = torch.zeros(window_size ** 2, window_size ** 2)
    mask = einops.rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
    mask[:, -displacement:, :, :-displacement] = float('-inf')
    mask[:, :-displacement, :, -displacement:] = float('-inf')
    mask = einops.rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')
    return mask


class WindowAttention(nn.Module):
    def __init__(self, window_size: int, embedding_dims: int, head_dims: int, shift: bool):
        super().__init__()
        assert 0 == (embedding_dims % head_dims)

        self.window_size = window_size
        self.embedding_dims = embedding_dims
        self.num_heads = embedding_dims // head_dims
        self.shift = shift

        if self.shift:
            self.register_buffer("upper_lower_mask", create_upper_lower_mask(window_size, window_size // 2))
            self.register_buffer("left_right_mask", create_left_right_mask(window_size, window_size // 2))

        self.fc_qkv = nn.Linear(self.embedding_dims, 3 * self.embedding_dims)
        self.attention_dropout = nn.Dropout(0.0)

        self.fc_proj = nn.Linear(self.embedding_dims, self.embedding_dims)
        self.output_dropout = nn.Dropout(0.0)

        self.rel_pos_enc = RelativePositionEncoding(self.num_heads, self.window_size, self.window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.window_size
        batch_dim, height, width, embedding_channels = x.size()
        assert embedding_channels == self.embedding_dims

        if self.shift:
            x = x.roll(shifts=(-w // 2, -w // 2), dims=(1, 2))

        # L == w*w
        x = einops.rearrange(x, 'b (h s1) (w s2) c -> b h w (s1 s2) c', s1=w, s2=w)  # (B H W C) -> (B h w L, C)

        q, k, v = self.fc_qkv(x).split(self.embedding_dims, dim=4)

        # (B, h, w, L, C) -> (B, h, w, L, H, c)  -> (B, h, w, H, L, c)
        q = q.view(batch_dim, height // w, width // w, w * w, self.num_heads,
                   self.embedding_dims // self.num_heads).transpose(3, 4)
        k = k.view(batch_dim, height // w, width // w, w * w, self.num_heads,
                   self.embedding_dims // self.num_heads).transpose(3, 4)
        v = v.view(batch_dim, height // w, width // w, w * w, self.num_heads,
                   self.embedding_dims // self.num_heads).transpose(3, 4)

        # -> (B, h, w, H, L, L)
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attention += self.rel_pos_enc.get_encoding()

        if self.shift:
            attention[:, -1, :, :, :, :] += self.upper_lower_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            attention[:, :, -1, :, :, :] += self.left_right_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)
        output = attention @ v

        output = output.transpose(3, 4).contiguous().view(batch_dim, height // w, width // w,
                                                          w * w * embedding_channels)
        output = einops.rearrange(output, 'b h w (s1 s2 c) -> b (h s1) (w s2) c', s1=w, s2=w)

        if self.shift:
            output = output.roll(shifts=(w // 2, w // 2), dims=(1, 2))

        output = self.output_dropout(self.fc_proj(output))

        return output


class PatchMerging(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels * 2 * 2)
        self.linear = nn.Linear(in_channels * 2 * 2, in_channels * 2, bias=False)

    def forward(self, x):
        x = einops.rearrange(x, 'b (h s1) (w s2) c -> b h w (s1 s2 c)', s1=2, s2=2)
        x = self.norm(x)
        x = self.linear(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_shape: Tuple[int, int, int],
                 patch_size: int,
                 embedding_dims: int,
                 depths: List[int],
                 window_sizes: List[int],
                 head_dims: int
                 ):
        super().__init__()

        self.in_shape = in_shape
        self.patch_size = patch_size
        self.embedding_dims = embedding_dims

        self.scale_to_depth = nn.PixelUnshuffle(self.patch_size)
        self.patch_embedding = nn.Linear(self.in_shape[0] * self.patch_size ** 2, self.embedding_dims)
        self.patch_norm = nn.LayerNorm(self.embedding_dims)
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.in_shape[1] // self.patch_size, self.in_shape[2] // self.patch_size,
                        self.embedding_dims).normal_(std=0.02),
            requires_grad=True)
        self.embedding_dropout = nn.Dropout(0.0)

        blocks = []
        channels = self.embedding_dims
        spatial_size = (in_shape[1] // patch_size, in_shape[2] // patch_size)
        for d_ix in range(len(depths)):
            if d_ix > 0:
                blocks.append(PatchMerging(channels))
                channels *= 2
                spatial_size = (spatial_size[0] // 2, spatial_size[1] // 2)

            single_window = spatial_size[0] == window_sizes[d_ix] and spatial_size[1] == window_sizes[d_ix]

            assert 0 == spatial_size[0] % window_sizes[d_ix] and 0 == spatial_size[1] % window_sizes[d_ix]
            assert single_window or (spatial_size[0] > window_sizes[d_ix] and spatial_size[1] > window_sizes[d_ix])

            for ix in range(depths[d_ix]):
                shift = (1 == ix % 2) and not single_window
                blocks.append(TransformerEncoder(WindowAttention(window_sizes[d_ix], channels, head_dims, shift),
                                                 channels))
        self.blocks = nn.Sequential(*blocks)

        self.output_norm = nn.LayerNorm(channels)
        self.output_activation = nn.GELU()
        self.output_dropout = nn.Dropout(0.3)

        self.classification = nn.Linear(channels, num_classes)

        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scale_to_depth(x)
        x = x.permute(0, 2, 3, 1)

        x = self.patch_embedding(x)
        x = self.patch_norm(x)
        x += self.position_embedding

        x = self.embedding_dropout(x)

        x = self.blocks(x)

        x = self.output_norm(x)
        x = self.output_activation(x)
        x = x.mean(dim=[1, 2])
        x = self.output_dropout(x)
        x = self.classification(x)

        return x

    def separate_parameters(self):
        parameters_with_decay = set()
        for m_name, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                for p_name, param in m.named_parameters():
                    if p_name.endswith("weight"):
                        parameters_with_decay.add(param)

        parameters_without_decay = set(self.parameters()) - parameters_with_decay

        # sanity check
        assert len(parameters_with_decay & parameters_without_decay) == 0
        assert len(parameters_with_decay) + len(parameters_without_decay) == len(list(self.parameters()))

        return list(parameters_with_decay), list(parameters_without_decay)
