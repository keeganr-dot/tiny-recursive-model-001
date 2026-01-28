"""Complete TRM network assembling all components."""
import torch
import torch.nn as nn

from .embedding import GridEmbedding
from .transformer import TRMStack
from .heads import OutputHead, HaltingHead


class TRMNetwork(nn.Module):
    """
    Complete TRM (Tiny Recursive Model) network.

    Assembles all components into end-to-end architecture:
    1. GridEmbedding: Maps ARC grids to continuous vectors
    2. TRMStack: 2-layer transformer with self-attention and SwiGLU
    3. OutputHead: Predicts color logits per grid cell
    4. HaltingHead: Predicts confidence for early stopping

    Architecture matches paper specification with ~7M parameters.

    Args:
        hidden_dim: Model dimension (default 512)
        num_heads: Number of attention heads (default 8)
        num_layers: Number of transformer layers (default 2)
        num_colors: Number of ARC color classes (default 10)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        num_colors: int = 10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_colors = num_colors

        # Component modules
        self.embedding = GridEmbedding(hidden_dim)
        self.transformer = TRMStack(hidden_dim, num_heads, num_layers)
        self.output_head = OutputHead(hidden_dim, num_colors)
        self.halting_head = HaltingHead(hidden_dim)

    def forward(self, grid: torch.Tensor) -> dict:
        """
        Process ARC grid through full network.

        Args:
            grid: Grid tensor of shape (B, H, W) with values 0-9 or -1 (padding)

        Returns:
            Dictionary containing:
                - logits: Color logits of shape (B, H, W, num_colors)
                - halt_confidence: Confidence scores of shape (B,) in range [0, 1]
        """
        B, H, W = grid.shape

        # Embed grid: (B, H, W) -> (B, H, W, hidden_dim)
        x = self.embedding(grid)

        # Flatten spatial dimensions to sequence: (B, H*W, hidden_dim)
        x = x.view(B, H * W, self.hidden_dim)

        # Apply transformer stack
        x = self.transformer(x)

        # Output head: (B, H*W, hidden_dim) -> (B, H*W, num_colors)
        logits = self.output_head(x)

        # Reshape logits back to grid: (B, H, W, num_colors)
        logits = logits.view(B, H, W, self.num_colors)

        # Halting head: (B, H*W, hidden_dim) -> (B,)
        halt_confidence = self.halting_head(x)

        return {
            "logits": logits,
            "halt_confidence": halt_confidence,
        }

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        Returns:
            Number of parameters in the network
        """
        return sum(p.numel() for p in self.parameters())
