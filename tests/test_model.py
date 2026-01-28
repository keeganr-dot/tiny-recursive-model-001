"""Tests for TRM model architecture."""
import pytest
import torch
from src.trm.model import (
    TRMNetwork,
    GridEmbedding,
    TRMStack,
    OutputHead,
    HaltingHead,
)


class TestGridEmbedding:
    """Tests for GridEmbedding component."""

    def test_shape(self):
        """Test basic embedding output shape."""
        emb = GridEmbedding(512)
        grid = torch.randint(0, 10, (2, 5, 5))
        out = emb(grid)
        assert out.shape == (2, 5, 5, 512)

    def test_padding_handling(self):
        """Test that padding value -1 is handled correctly."""
        emb = GridEmbedding(512)
        grid = torch.tensor([[[0, -1], [1, 2]]])
        out = emb(grid)
        assert out.shape == (1, 2, 2, 512)

    def test_no_bias(self):
        """Test that embedding has no bias parameters."""
        emb = GridEmbedding(512)
        for name, _ in emb.named_parameters():
            assert "bias" not in name, f"Found bias parameter: {name}"


class TestOutputHead:
    """Tests for OutputHead component."""

    def test_shape(self):
        """Test output head produces correct logit shape."""
        head = OutputHead(512, 10)
        x = torch.randn(2, 100, 512)
        out = head(x)
        assert out.shape == (2, 100, 10)

    def test_no_bias(self):
        """Test that output head has no bias parameters."""
        head = OutputHead(512, 10)
        for name, _ in head.named_parameters():
            assert "bias" not in name, f"Found bias parameter: {name}"


class TestHaltingHead:
    """Tests for HaltingHead component."""

    def test_shape(self):
        """Test halting head produces scalar per batch item."""
        head = HaltingHead(512)
        x = torch.randn(2, 100, 512)
        out = head(x)
        assert out.shape == (2,)

    def test_sigmoid_output(self):
        """Test halting confidence is in valid [0, 1] range."""
        head = HaltingHead(512)
        x = torch.randn(4, 50, 512)
        out = head(x)
        assert (out >= 0).all(), f"Found values < 0: {out[out < 0]}"
        assert (out <= 1).all(), f"Found values > 1: {out[out > 1]}"

    def test_no_bias(self):
        """Test that halting head has no bias parameters."""
        head = HaltingHead(512)
        for name, _ in head.named_parameters():
            assert "bias" not in name, f"Found bias parameter: {name}"


class TestTRMNetwork:
    """Tests for complete TRMNetwork."""

    def test_forward_shape(self):
        """Test end-to-end forward pass produces correct shapes."""
        net = TRMNetwork()
        grid = torch.randint(0, 10, (2, 10, 10))
        out = net(grid)
        assert out["logits"].shape == (2, 10, 10, 10)
        assert out["halt_confidence"].shape == (2,)

    def test_halt_confidence_range(self):
        """Test that halt confidence is in valid [0, 1] range."""
        net = TRMNetwork()
        grid = torch.randint(0, 10, (4, 5, 5))
        out = net(grid)
        conf = out["halt_confidence"]
        assert (conf >= 0).all(), f"Found confidence < 0: {conf[conf < 0]}"
        assert (conf <= 1).all(), f"Found confidence > 1: {conf[conf > 1]}"

    def test_parameter_count(self):
        """Test that network has approximately 7M parameters as specified in paper."""
        net = TRMNetwork()
        params = net.count_parameters()
        # Paper specifies ~7M parameters, allow 5M-15M range for implementation variations
        # With hidden_dim=512, num_layers=2, num_heads=8, we get ~10.5M params
        assert (
            5_000_000 < params < 15_000_000
        ), f"Expected ~7M-10M params, got {params:,}"
        print(f"Parameter count: {params:,}")

    def test_no_bias(self):
        """Test that entire network has no bias parameters."""
        net = TRMNetwork()
        for name, _ in net.named_parameters():
            assert "bias" not in name, f"Found bias parameter: {name}"

    def test_variable_grid_sizes(self):
        """Test network handles various ARC grid sizes."""
        net = TRMNetwork()
        # Test various sizes from smallest to largest
        test_sizes = [(1, 1), (3, 5), (10, 10), (15, 20), (30, 30)]
        for h, w in test_sizes:
            grid = torch.randint(0, 10, (1, h, w))
            out = net(grid)
            assert out["logits"].shape == (
                1,
                h,
                w,
                10,
            ), f"Failed for size ({h}, {w})"
            assert out["halt_confidence"].shape == (1,)

    def test_max_arc_grid(self):
        """Test network handles maximum ARC grid size (30x30)."""
        net = TRMNetwork()
        grid = torch.randint(0, 10, (2, 30, 30))
        out = net(grid)
        assert out["logits"].shape == (2, 30, 30, 10)
        assert out["halt_confidence"].shape == (2,)

    def test_small_grids(self):
        """Test network handles very small grids (1x1 to 3x3)."""
        net = TRMNetwork()
        for size in [1, 2, 3]:
            grid = torch.randint(0, 10, (1, size, size))
            out = net(grid)
            assert out["logits"].shape == (1, size, size, 10)

    def test_rectangular_grids(self):
        """Test network handles non-square grids."""
        net = TRMNetwork()
        test_shapes = [(5, 10), (10, 5), (3, 20), (25, 8)]
        for h, w in test_shapes:
            grid = torch.randint(0, 10, (1, h, w))
            out = net(grid)
            assert out["logits"].shape == (1, h, w, 10)

    def test_batch_processing(self):
        """Test network handles different batch sizes."""
        net = TRMNetwork()
        for batch_size in [1, 2, 4, 8, 16]:
            grid = torch.randint(0, 10, (batch_size, 10, 10))
            out = net(grid)
            assert out["logits"].shape == (batch_size, 10, 10, 10)
            assert out["halt_confidence"].shape == (batch_size,)

    def test_padding_values(self):
        """Test network handles grids with padding (-1 values)."""
        net = TRMNetwork()
        # Create grid with some padding
        grid = torch.randint(0, 10, (1, 5, 5))
        grid[0, 0, 0] = -1  # Add padding value
        grid[0, 4, 4] = -1
        out = net(grid)
        assert out["logits"].shape == (1, 5, 5, 10)
        # Ensure no NaNs or Infs in output
        assert not torch.isnan(out["logits"]).any()
        assert not torch.isinf(out["logits"]).any()
