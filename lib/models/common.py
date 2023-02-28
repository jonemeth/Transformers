import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, attention: nn.Module, embedding_dims: int):
        super().__init__()

        self.attention = attention

        self.norm1 = nn.LayerNorm(embedding_dims)
        self.norm2 = nn.LayerNorm(embedding_dims)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dims, 4 * embedding_dims),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(4 * embedding_dims, embedding_dims),
            nn.Dropout(0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.attention(self.norm1(x))
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class RelativePositionEncoding(nn.Module):
    def __init__(self, num_heads: int, height: int, width: int):
        super().__init__()

        self.num_heads = num_heads
        self.height = height
        self.width = width

        self.pos_enc = nn.Parameter(
            torch.empty(num_heads, (2 * height - 1) * (2 * width - 1)).normal_(std=0.02))

        self.register_buffer("relative_indices", self.get_indices())

    def get_encoding(self):
        rel_pos_enc = self.pos_enc.gather(-1, self.relative_indices)
        rel_pos_enc = rel_pos_enc.unflatten(-1, (self.height*self.width, self.height*self.width))
        return rel_pos_enc

    def get_indices(self):
        h, w = self.height, self.width

        y = torch.arange(h, dtype=torch.long)
        x = torch.arange(w, dtype=torch.long)

        y1, x1, y2, x2 = torch.meshgrid(y, x, y, x, indexing='ij')
        indices = (y1 - y2 + h - 1) * (2 * w - 1) + x1 - x2 + w - 1
        indices = indices.flatten()

        return indices.expand(self.num_heads, -1)
