"""Transformer layer and stack for TRM architecture."""
import torch
import torch.nn as nn

from .attention import Attention
from .layers import RMSNorm, SwiGLU


class TRMLayer(nn.Module):
    """
    Single transformer layer with pre-norm architecture.

    Architecture:
    1. RMSNorm -> Self-Attention -> Residual
    2. RMSNorm -> SwiGLU FFN -> Residual

    Uses SwiGLU for feed-forward network (not standard MLP).
    No bias in any component per paper specification.

    Args:
        hidden_dim: Model dimension (default 512)
        num_heads: Number of attention heads (default 8)
    """

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Pre-norm for attention
        self.attn_norm = RMSNorm(hidden_dim)
        self.attn = Attention(hidden_dim, num_heads)

        # Pre-norm for FFN
        self.ffn_norm = RMSNorm(hidden_dim)
        self.ffn = SwiGLU(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply transformer layer.

        Args:
            x: Input tensor of shape (B, seq_len, hidden_dim)
            mask: Optional attention mask of shape (B, seq_len) or (B, seq_len, seq_len)
            positions: Optional position indices of shape (B, seq_len)

        Returns:
            Output tensor of shape (B, seq_len, hidden_dim)
        """
        # Pre-norm attention with residual
        x = x + self.attn(self.attn_norm(x), mask=mask, positions=positions)

        # Pre-norm FFN with residual
        x = x + self.ffn(self.ffn_norm(x))

        return x


class TRMStack(nn.Module):
    """
    Stack of transformer layers for TRM architecture.

    Per paper: 2-layer network with weight sharing in recursive loop (Phase 4).
    This module creates the base 2-layer stack without weight sharing.

    Args:
        hidden_dim: Model dimension (default 512)
        num_heads: Number of attention heads (default 8)
        num_layers: Number of transformer layers (default 2)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Create stack of layers
        self.layers = nn.ModuleList([
            TRMLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply all transformer layers sequentially.

        Args:
            x: Input tensor of shape (B, seq_len, hidden_dim)
            mask: Optional attention mask of shape (B, seq_len) or (B, seq_len, seq_len)
            positions: Optional position indices of shape (B, seq_len)

        Returns:
            Output tensor of shape (B, seq_len, hidden_dim)
        """
        for layer in self.layers:
            x = layer(x, mask=mask, positions=positions)

        return x
