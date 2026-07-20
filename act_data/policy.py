"""A compact ACT-style policy for chunked action prediction."""

from __future__ import annotations

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """Small CNN encoder shared across camera views."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class ACTPolicy(nn.Module):
    """Image + qpos conditioned transformer decoder with action queries."""

    def __init__(
        self,
        *,
        qpos_dim: int,
        action_dim: int,
        num_cameras: int,
        chunk_size: int,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.qpos_dim = int(qpos_dim)
        self.action_dim = int(action_dim)
        self.num_cameras = int(num_cameras)
        self.chunk_size = int(chunk_size)
        self.hidden_dim = int(hidden_dim)

        self.image_encoder = ImageEncoder(hidden_dim)
        self.qpos_proj = nn.Sequential(
            nn.Linear(qpos_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.source_type_embed = nn.Embedding(num_cameras + 1, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.action_queries = nn.Parameter(torch.randn(chunk_size, hidden_dim) * 0.02)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self._config = {
            "qpos_dim": self.qpos_dim,
            "action_dim": self.action_dim,
            "num_cameras": self.num_cameras,
            "chunk_size": self.chunk_size,
            "hidden_dim": self.hidden_dim,
            "nheads": int(nheads),
            "num_encoder_layers": int(num_encoder_layers),
            "num_decoder_layers": int(num_decoder_layers),
            "dropout": float(dropout),
        }

    def export_config(self) -> dict:
        return dict(self._config)

    def forward(self, image: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        """Predict action chunks.

        Args:
            image: Float tensor shaped [B, num_cameras, 3, H, W].
            qpos: Float tensor shaped [B, qpos_dim].
        Returns:
            Float tensor shaped [B, chunk_size, action_dim].
        """

        if image.ndim != 5:
            raise ValueError(f"image must be [B,N,3,H,W], got {tuple(image.shape)}")
        batch, num_cameras, channels, height, width = image.shape
        if num_cameras != self.num_cameras:
            raise ValueError(
                f"expected {self.num_cameras} cameras, got {num_cameras}"
            )
        if channels != 3:
            raise ValueError(f"expected RGB images, got {channels} channels")

        flat_images = image.reshape(batch * num_cameras, channels, height, width)
        camera_tokens = self.image_encoder(flat_images).reshape(
            batch, num_cameras, self.hidden_dim
        )
        qpos_token = self.qpos_proj(qpos).unsqueeze(1)
        memory = torch.cat([camera_tokens, qpos_token], dim=1)

        token_types = torch.arange(
            num_cameras + 1,
            device=image.device,
            dtype=torch.long,
        )
        memory = memory + self.source_type_embed(token_types).unsqueeze(0)
        memory = self.encoder(memory)

        queries = self.action_queries.unsqueeze(0).expand(batch, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
    """L1 loss over non-padded action chunk elements."""

    valid = (~is_pad).unsqueeze(-1).to(dtype=pred.dtype)
    denom = valid.sum().clamp_min(1.0) * pred.shape[-1]
    return (torch.abs(pred - target) * valid).sum() / denom
