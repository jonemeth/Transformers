import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.models.common import TransformerEncoder, RelativePositionEncoding


# https://juliusruseckas.github.io/ml/cifar10-vit.html


class SelfAttention(nn.Module):
    def __init__(self, shape2d: Tuple[int, int], embedding_dims: int, num_heads: int, relative_encoding: bool,
                 class_embedding: bool):
        super().__init__()
        assert 0 == (embedding_dims % num_heads)

        self.shape2d = shape2d
        self.embedding_dims = embedding_dims
        self.num_heads = num_heads
        self.relative_encoding = relative_encoding
        self.class_embedding = class_embedding

        self.fc_q = nn.Linear(self.embedding_dims, self.embedding_dims)
        self.fc_k = nn.Linear(self.embedding_dims, self.embedding_dims)
        self.fc_v = nn.Linear(self.embedding_dims, self.embedding_dims)
        self.attention_dropout = nn.Dropout(0.0)

        self.fc_proj = nn.Linear(self.embedding_dims, self.embedding_dims)
        self.output_dropout = nn.Dropout(0.0)

        if self.relative_encoding:
            height, width = self.shape2d
            self.rel_pos_enc = RelativePositionEncoding(self.num_heads, height, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_dim, num_embeddings, embedding_channels = x.size()

        q, k, v = self.fc_q(x), self.fc_k(x), self.fc_v(x)

        # (B, H, L, HC)
        q = q.view(batch_dim, num_embeddings, self.num_heads, self.embedding_dims // self.num_heads).transpose(1, 2)
        k = k.view(batch_dim, num_embeddings, self.num_heads, self.embedding_dims // self.num_heads).transpose(1, 2)
        v = v.view(batch_dim, num_embeddings, self.num_heads, self.embedding_dims // self.num_heads).transpose(1, 2)

        # (B, H, L, HC) x (B, H, HC, L) -> (B, H, L, L)
        attention = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.relative_encoding:
            if self.class_embedding:
                attention[:, :, 1:, 1:] += self.rel_pos_enc.get_encoding()
            else:
                attention += self.rel_pos_enc.get_encoding()

        attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(attention)

        output = attention @ v  # (B, H, L, L) x (B, H, L, HC) -> (B, H, L, HC)
        output = output.transpose(1, 2).contiguous().view(batch_dim, num_embeddings, embedding_channels)  # (B, L, C)

        output = self.output_dropout(self.fc_proj(output))
        return output

    @staticmethod
    def get_indices(h, w):
        y = torch.arange(h, dtype=torch.long)
        x = torch.arange(w, dtype=torch.long)

        y1, x1, y2, x2 = torch.meshgrid(y, x, y, x, indexing='ij')
        indices = (y1 - y2 + h - 1) * (2 * w - 1) + x1 - x2 + w - 1
        indices = indices.flatten()

        return indices


class SimplePatches(nn.Module):
    def __init__(self, in_shape, channels, patch_size):
        super().__init__()
        self.in_shape = in_shape
        self.patch_size = patch_size
        self.num_patches = (self.in_shape[1] // self.patch_size) * (self.in_shape[2] // self.patch_size)
        self.scale_to_depth = nn.PixelUnshuffle(self.patch_size)
        self.patch_embedding = nn.Linear(in_shape[0] * patch_size ** 2, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.scale_to_depth(x)
        x = x.view(x.shape[0], self.in_shape[0] * self.patch_size ** 2, self.num_patches).transpose(1, 2)
        x = self.patch_embedding(x)
        return x


class ConvPatches(nn.Module):
    def __init__(self, in_shape, channels, patch_size, hidden_channels=32):
        super().__init__()
        self.in_shape = in_shape
        self.patch_size = patch_size
        self.channels = channels
        self.num_patches = (self.in_shape[1] // self.patch_size) * (self.in_shape[2] // self.patch_size)

        self.conv0 = nn.Conv2d(in_shape[0], hidden_channels, 3, padding=1)
        self.act0 = nn.GELU()
        self.conv1 = nn.Conv2d(hidden_channels, channels, patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act0(self.conv0(x))
        x = self.conv1(x)
        x = x.view(x.shape[0], self.channels, self.num_patches).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, in_shape: Tuple[int, int, int], patch_size: int,
                 embedding_dims: int, depth: int, num_heads: int, num_classes: int,
                 conv_patches: bool, relative_encoding: bool, avg_pool: bool):
        super().__init__()

        self.in_shape = in_shape
        self.patch_size = patch_size
        self.embedding_dims = embedding_dims

        if conv_patches:
            self.to_patches = ConvPatches(self.in_shape, self.embedding_dims, self.patch_size)
        else:
            self.to_patches = SimplePatches(self.in_shape, self.embedding_dims, self.patch_size)

        self.num_patches = (self.in_shape[1] // self.patch_size) * (self.in_shape[2] // self.patch_size)

        self.position_embedding = nn.Parameter(torch.empty(1, self.num_patches,
                                                           self.embedding_dims).normal_(std=0.02),
                                               requires_grad=True)

        self.class_embedding = nn.Parameter(torch.empty(1, 1, self.embedding_dims).normal_(std=0.02),
                                            requires_grad=True) if not avg_pool else None

        self.embedding_dropout = nn.Dropout(0.0)

        shape2d = (in_shape[1] // patch_size, in_shape[2] // patch_size)
        self.blocks = nn.Sequential(
            *[TransformerEncoder(SelfAttention(shape2d, self.embedding_dims, num_heads, relative_encoding,
                                               not avg_pool), self.embedding_dims) for _ in range(depth)])

        self.output_norm = nn.LayerNorm(embedding_dims)
        self.output_act = nn.GELU()

        self.classification_dropout = nn.Dropout(0.3)
        self.classification = nn.Linear(self.embedding_dims, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.to_patches(x)
        x += self.position_embedding

        if self.class_embedding is not None:
            x = torch.cat([self.class_embedding.repeat(x.shape[0], 1, 1), x], dim=1)  # Prepend the class embedding

        x = self.embedding_dropout(x)
        x = self.blocks(x)

        if self.class_embedding is not None:
            x = x[:, 0, :]
            x = self.output_act(self.output_norm(x))
        else:
            x = self.output_act(self.output_norm(x))
            x = x.mean(1, keepdim=False)

        x = self.classification_dropout(x)
        c = self.classification(x)

        return c

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
