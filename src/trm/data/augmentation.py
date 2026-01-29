"""Data augmentation for ARC-AGI grids.

This module provides geometric (D8 dihedral group) and color permutation
augmentations to increase effective training data size.

D8 dihedral group: The 8 symmetries of a square (4 rotations + 4 reflections)
Color permutations: Shuffling the 10 ARC colors (0-9) while preserving structure
"""
import torch
from typing import List, Tuple, Optional
import random


class D8Transform:
    """D8 dihedral group transformations (8 symmetries of a square).

    The D8 group consists of:
    - 4 rotations: 0°, 90°, 180°, 270°
    - 4 reflections: horizontal, vertical, main diagonal, anti-diagonal

    All transformations preserve the grid structure and can be applied
    consistently to input-output pairs for training.
    """

    def __init__(self, transform_id: int):
        """Initialize a D8 transform.

        Args:
            transform_id: Integer in [0-7] identifying the transformation:
                0: Identity (no change)
                1: Rotate 90° clockwise
                2: Rotate 180°
                3: Rotate 270° clockwise (90° counter-clockwise)
                4: Flip horizontal (left-right)
                5: Flip vertical (top-bottom)
                6: Flip along main diagonal (transpose)
                7: Flip along anti-diagonal (transpose + 180° rotation)
        """
        if not 0 <= transform_id <= 7:
            raise ValueError(f"transform_id must be in [0, 7], got {transform_id}")
        self.transform_id = transform_id

    @staticmethod
    def get_all_transforms() -> List['D8Transform']:
        """Get all 8 D8 group transformations.

        Returns:
            List of 8 D8Transform instances (one for each group element)
        """
        return [D8Transform(i) for i in range(8)]

    def apply(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply the D8 transformation to a grid.

        Args:
            grid: Tensor of shape (H, W) or (B, H, W) containing grid values

        Returns:
            Transformed grid with same dtype and device as input
        """
        if self.transform_id == 0:
            # Identity
            return grid.clone()
        elif self.transform_id == 1:
            # Rotate 90° clockwise
            return torch.rot90(grid, k=1, dims=(-2, -1))
        elif self.transform_id == 2:
            # Rotate 180°
            return torch.rot90(grid, k=2, dims=(-2, -1))
        elif self.transform_id == 3:
            # Rotate 270° clockwise
            return torch.rot90(grid, k=3, dims=(-2, -1))
        elif self.transform_id == 4:
            # Flip horizontal (left-right)
            return torch.flip(grid, dims=[-1])
        elif self.transform_id == 5:
            # Flip vertical (top-bottom)
            return torch.flip(grid, dims=[-2])
        elif self.transform_id == 6:
            # Transpose (main diagonal flip)
            return grid.transpose(-2, -1)
        elif self.transform_id == 7:
            # Anti-diagonal flip (transpose + 180° rotation)
            return torch.rot90(grid.transpose(-2, -1), k=2, dims=(-2, -1))
        else:
            raise ValueError(f"Invalid transform_id: {self.transform_id}")

    def apply_pair(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the SAME transformation to both input and output grids.

        This ensures geometric consistency required for training - the input-output
        relationship is preserved under the transformation.

        Args:
            input_grid: Input grid tensor
            output_grid: Output grid tensor

        Returns:
            Tuple of (transformed_input, transformed_output)
        """
        return self.apply(input_grid), self.apply(output_grid)

    def __repr__(self) -> str:
        names = ["Identity", "Rot90", "Rot180", "Rot270",
                 "FlipH", "FlipV", "Transpose", "AntiDiag"]
        return f"D8Transform({self.transform_id}: {names[self.transform_id]})"


class ColorPermutation:
    """Color permutation augmentation for ARC grids.

    Shuffles the 10 ARC colors (0-9) while preserving grid structure.
    This creates up to 10! = 3,628,800 possible augmentations per grid.

    Important: Handles PAD_VALUE (-1) correctly by leaving it unchanged.
    """

    def __init__(self, permutation: Optional[torch.Tensor] = None):
        """Initialize a color permutation.

        Args:
            permutation: Tensor of shape (10,) mapping old colors to new colors.
                        If None, generates a random permutation.
        """
        if permutation is None:
            permutation = torch.randperm(10)

        if permutation.shape != (10,):
            raise ValueError(f"Permutation must have shape (10,), got {permutation.shape}")

        self.permutation = permutation

    @staticmethod
    def random() -> 'ColorPermutation':
        """Create a random color permutation.

        Returns:
            ColorPermutation with randomly shuffled colors
        """
        return ColorPermutation(torch.randperm(10))

    def apply(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply color permutation to a grid.

        Args:
            grid: Tensor containing color values (0-9) and possibly PAD_VALUE (-1)

        Returns:
            Grid with permuted colors, PAD_VALUE preserved
        """
        # Handle padding: -1 should remain -1
        pad_mask = (grid == -1)

        # Clamp to valid range [0, 9] for indexing
        clamped = grid.clamp(min=0)

        # Apply permutation
        result = self.permutation[clamped]

        # Restore padding
        result[pad_mask] = -1

        return result

    def apply_pair(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the SAME color permutation to both input and output grids.

        This ensures color consistency - if color A maps to color B in the input,
        it also maps to color B in the output.

        Args:
            input_grid: Input grid tensor
            output_grid: Output grid tensor

        Returns:
            Tuple of (permuted_input, permuted_output)
        """
        return self.apply(input_grid), self.apply(output_grid)

    @staticmethod
    def compose(perm1: 'ColorPermutation', perm2: 'ColorPermutation') -> 'ColorPermutation':
        """Compose two color permutations.

        The result applies perm1 first, then perm2.

        Args:
            perm1: First permutation to apply
            perm2: Second permutation to apply

        Returns:
            Composed permutation
        """
        composed = perm2.permutation[perm1.permutation]
        return ColorPermutation(composed)

    def __repr__(self) -> str:
        return f"ColorPermutation({self.permutation.tolist()})"


class AugmentationPipeline:
    """Combined augmentation pipeline for D8 and color permutations.

    Applies geometric and/or color augmentations to input-output grid pairs,
    ensuring consistency between input and output transformations.
    """

    def __init__(self, enable_d8: bool = True, enable_color: bool = True):
        """Initialize augmentation pipeline.

        Args:
            enable_d8: Whether to apply D8 geometric transforms
            enable_color: Whether to apply color permutations
        """
        self.enable_d8 = enable_d8
        self.enable_color = enable_color

    def augment_pair(self, input_grid: torch.Tensor, output_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply augmentations to an input-output pair.

        Args:
            input_grid: Input grid tensor
            output_grid: Output grid tensor

        Returns:
            Tuple of (augmented_input, augmented_output) with consistent transforms
        """
        aug_input, aug_output = input_grid, output_grid

        # Apply D8 transform if enabled
        if self.enable_d8:
            transform = D8Transform(random.randint(0, 7))
            aug_input, aug_output = transform.apply_pair(aug_input, aug_output)

        # Apply color permutation if enabled
        if self.enable_color:
            permutation = ColorPermutation.random()
            aug_input, aug_output = permutation.apply_pair(aug_input, aug_output)

        return aug_input, aug_output

    def get_effective_multiplier(self) -> int:
        """Get the effective data multiplier from enabled augmentations.

        Returns:
            Integer multiplier:
                - 1 if nothing enabled
                - 8 if only D8 enabled
                - 3628800 if only color enabled (10!)
                - 29030400 if both enabled (8 * 10!)
        """
        multiplier = 1

        if self.enable_d8:
            multiplier *= 8

        if self.enable_color:
            multiplier *= 3628800  # 10!

        return multiplier

    def __repr__(self) -> str:
        return f"AugmentationPipeline(d8={self.enable_d8}, color={self.enable_color}, multiplier={self.get_effective_multiplier()})"


if __name__ == "__main__":
    print("Testing D8 transforms...")

    # Create a test grid with distinct values
    test_grid = torch.tensor([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    print(f"Original grid:\n{test_grid}\n")

    # Apply all 8 transforms
    transforms = D8Transform.get_all_transforms()
    results = []

    for t in transforms:
        result = t.apply(test_grid)
        results.append(result)
        print(f"{t}:\n{result}\n")

    # Verify we get 8 distinct results
    unique_results = []
    for r in results:
        is_unique = True
        for u in unique_results:
            if torch.equal(r, u):
                is_unique = False
                break
        if is_unique:
            unique_results.append(r)

    assert len(unique_results) == 8, f"Expected 8 distinct transforms, got {len(unique_results)}"

    # Verify identity transform
    identity = D8Transform(0).apply(test_grid)
    assert torch.equal(identity, test_grid), "Identity transform should return unchanged grid"

    print("[PASS] D8 transforms verified: 8 distinct orientations\n")

    # Test color permutation
    print("Testing color permutation...")

    color_grid = torch.tensor([
        [0, 1, 2],
        [3, 4, 5],
        [-1, -1, -1]  # Padding row
    ])

    print(f"Original grid:\n{color_grid}\n")

    perm = ColorPermutation.random()
    permuted = perm.apply(color_grid)

    print(f"Permuted grid:\n{permuted}\n")
    print(f"Permutation: {perm.permutation.tolist()}\n")

    # Verify padding is preserved
    assert torch.all(permuted[2, :] == -1), "PAD_VALUE (-1) should be preserved"

    # Verify structure is preserved (same positions are non-padding)
    assert (color_grid != -1).sum() == (permuted != -1).sum(), "Structure should be preserved"

    # Test apply_pair consistency
    input_grid = torch.tensor([[0, 1], [2, 3]])
    output_grid = torch.tensor([[4, 5], [6, 7]])

    perm2 = ColorPermutation(torch.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
    perm_in, perm_out = perm2.apply_pair(input_grid, output_grid)

    # Verify same permutation applied to both
    assert torch.equal(perm_in, perm2.apply(input_grid)), "apply_pair should use same permutation"
    assert torch.equal(perm_out, perm2.apply(output_grid)), "apply_pair should use same permutation"

    print("[PASS] Color permutation verified: structure preserved, PAD_VALUE handled\n")

    # Test combined pipeline
    print("Testing augmentation pipeline...")

    pipeline = AugmentationPipeline(enable_d8=True, enable_color=True)
    print(f"{pipeline}\n")

    test_in = torch.tensor([[0, 1], [2, 3]])
    test_out = torch.tensor([[1, 2], [3, 4]])

    aug_in, aug_out = pipeline.augment_pair(test_in, test_out)

    print(f"Input: {test_in.tolist()} -> {aug_in.tolist()}")
    print(f"Output: {test_out.tolist()} -> {aug_out.tolist()}")

    # Verify effective multiplier calculation
    assert pipeline.get_effective_multiplier() == 8 * 3628800, "Wrong multiplier calculation"

    pipeline_d8_only = AugmentationPipeline(enable_d8=True, enable_color=False)
    assert pipeline_d8_only.get_effective_multiplier() == 8, "D8-only should be 8x"

    pipeline_color_only = AugmentationPipeline(enable_d8=False, enable_color=True)
    assert pipeline_color_only.get_effective_multiplier() == 3628800, "Color-only should be 10!"

    pipeline_disabled = AugmentationPipeline(enable_d8=False, enable_color=False)
    assert pipeline_disabled.get_effective_multiplier() == 1, "Disabled should be 1x"

    print("\n[PASS] Augmentation pipeline verified")
    print("\n" + "="*50)
    print("ALL TESTS PASSED")
    print("="*50)
