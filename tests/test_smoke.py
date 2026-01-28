"""Smoke tests for environment verification."""
import sys


def test_python_version():
    """Verify Python 3.10+."""
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"


def test_torch_import():
    """Verify PyTorch installation."""
    import torch
    assert torch.__version__ >= "2.0.0", f"PyTorch 2.0+ required, got {torch.__version__}"


def test_cuda_detection():
    """Check CUDA availability (informational, not required)."""
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"\nCUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("\nCUDA not available - will use CPU")
    # Don't assert - CPU training is valid


def test_einops_import():
    """Verify einops installation."""
    import einops
    from einops import rearrange, repeat
    # Quick functional test
    import torch
    x = torch.randn(2, 3, 4)
    y = rearrange(x, 'b h w -> b (h w)')
    assert y.shape == (2, 12)


def test_hydra_import():
    """Verify hydra-core installation."""
    import hydra
    from omegaconf import OmegaConf
    # Quick functional test
    cfg = OmegaConf.create({"key": "value"})
    assert cfg.key == "value"


def test_trm_package():
    """Verify TRM package imports."""
    from src.trm import __version__
    assert __version__ == "0.1.0"


if __name__ == "__main__":
    """Run all tests and report."""
    import pytest
    exit_code = pytest.main([__file__, "-v"])
    sys.exit(exit_code)
