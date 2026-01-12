# SwiGLU Activation Function Implementation
# Reference: https://arxiv.org/abs/2002.05202 (GLU-Variants Improve Transformer)
# Used in: LLaMA, Mistral, DeepSeek, ChatGLM, etc.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    """Model arguments for SwiGLU FFN layer."""
    dim: int = 4096          # Input dimension
    hidden_dim: int = None   # Hidden layer dimension (usually 2/3 * 4 * dim ≈ 2.67 * dim)
    multiple_of: int = 256   # Make hidden_dim a multiple of this for efficiency
    ffn_dim_multiplier: Optional[float] = None  # Optional multiplier for hidden_dim


class SwiGLU(nn.Module):
    """
    SwiGLU Activation Function.

    SwiGLU = (xW_g ⊙ SiLU(xW_1))W_2

    where:
        - SiLU(x) = x * sigmoid(x) (also called Swish)
        - ⊙ denotes element-wise multiplication
        - W_g, W_1 are projection matrices
        - W_2 is the output projection matrix

    This is a variant of the GLU (Gated Linear Unit) activation that uses
    SiLU (Swish) as the activation function. It has become the de facto
    standard for modern LLMs.

    Args:
        dim: Input dimension
        hidden_dim: Hidden layer dimension (if None, computed from dim and multiple_of)

    Shape:
        - Input: (batch_size, seq_len, dim)
        - Output: (batch_size, seq_len, dim)
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            # Default: 2/3 * 4 * dim ≈ 2.67 * dim (commonly used in LLaMA)
            hidden_dim = int((4 * dim) * 2 / 3)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)   # Gate branch
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)   # Output projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)   # Value branch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SwiGLU.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        # Compute gate: SiLU(xW_1)
        gate = F.silu(self.w1(x))

        # Compute value: xW_3
        value = self.w3(x)

        # Element-wise multiplication and output projection
        return self.w2(gate * value)


class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation (standard Transformer FFN).

    This is the complete FFN layer used in modern LLMs:
        FFN(x) = SwiGLU(x) = (xW_g ⊙ SiLU(xW_1))W_2

    Args:
        args: ModelArgs containing dimension configuration

    Example:
        >>> args = ModelArgs(dim=4096, multiple_of=256)
        >>> ffn = SwiGLUFFN(args)
        >>> x = torch.randn(2, 128, 4096)  # batch=2, seq_len=128, dim=4096
        >>> output = ffn(x)
        >>> print(output.shape)  # torch.Size([2, 128, 4096])
    """

    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.dim
        hidden_dim = args.hidden_dim

        # Compute hidden_dim if not specified
        if hidden_dim is None:
            multiple_of = args.multiple_of
            # LLaMA formula: 2/3 * 4 * dim
            hidden_dim = int((4 * dim) * 2 / 3)
            # Make it a multiple of `multiple_of` for efficient tensor core utilization
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        # Apply ffn_dim_multiplier if specified (for model scaling)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SwiGLU FFN.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        # SiLU(xW_1) ⊙ xW_3, then W_2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GeGLU(nn.Module):
    """
    GeGLU Activation Function (Gated Exponential Linear Unit).

    GeGLU = (xW_g ⊙ GeLU(xW_1))W_2

    This is another GLU variant that uses GELU instead of SiLU.
    Used in some models like PaLM and T5.

    Args:
        dim: Input dimension
        hidden_dim: Hidden layer dimension

    Note: SwiGLU generally outperforms GeGLU in practice, which is why
    most modern LLMs use SwiGLU instead.
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int((4 * dim) * 2 / 3)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GeGLU.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)

        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        # GELU(xW_1) ⊙ xW_3, then W_2
        return self.w2(F.gelu(self.w1(x)) * self.w3(x))


# Comparison with vanilla ReLU FFN
class VanillaFFN(nn.Module):
    """
    Traditional FFN with ReLU activation (for comparison).

    Vanilla FFN(x) = ReLU(xW_1)W_2

    This was the original activation used in Transformer and BERT.
    Modern LLMs have replaced this with SwiGLU for better performance.

    Args:
        dim: Input dimension
        hidden_dim: Hidden layer dimension (typically 4 * dim)
    """

    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)))


def swiglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Functional form of SwiGLU activation.

    Args:
        x: Value tensor
        gate: Gate tensor (same shape as x)

    Returns:
        SwiGLU(x, gate) = x * SiLU(gate)

    Example:
        >>> x = torch.randn(10, 100)
        >>> gate = torch.randn(10, 100)
        >>> output = swiglu(x, gate)
    """
    return x * F.silu(gate)


def geglu(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """
    Functional form of GeGLU activation.

    Args:
        x: Value tensor
        gate: Gate tensor (same shape as x)

    Returns:
        GeGLU(x, gate) = x * GELU(gate)
    """
    return x * F.gelu(gate)


if __name__ == "__main__":
    # Simple test and demonstration
    print("=" * 60)
    print("SwiGLU Activation Function - Test and Comparison")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    seq_len = 128
    dim = 512

    args = ModelArgs(dim=dim, multiple_of=256)

    # Create test input
    x = torch.randn(batch_size, seq_len, dim)
    print(f"\nInput shape: {x.shape}")

    # Test SwiGLU
    print("\n--- SwiGLU FFN ---")
    swiglu_ffn = SwiGLUFFN(args)
    print(f"Hidden dim: {args.hidden_dim}")
    output = swiglu_ffn(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in swiglu_ffn.parameters()):,}")

    # Test GeGLU
    print("\n--- GeGLU FFN ---")
    geglu_ffn = GeGLU(dim=dim)
    output = geglu_ffn(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in geglu_ffn.parameters()):,}")

    # Test Vanilla FFN (for comparison)
    print("\n--- Vanilla ReLU FFN (for comparison) ---")
    vanilla_ffn = VanillaFFN(dim=dim)
    output = vanilla_ffn(x)
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in vanilla_ffn.parameters()):,}")

    print("\n" + "=" * 60)
    print("SwiGLU vs Vanilla FFN Parameter Comparison:")
    print("=" * 60)
    print(f"Vanilla FFN (ReLU):  2 * d * 4d = {2 * dim * 4 * dim:,} parameters")
    print(f"SwiGLU FFN:           3 * d * 4d * 2/3 ≈ {sum(p.numel() for p in swiglu_ffn.parameters()):,} parameters")
    print(f"\nSwiGLU has ~1.5x more parameters but significantly better performance.")

    # Functional test
    print("\n--- Functional API Test ---")
    gate = torch.randn(10, 100)
    value = torch.randn(10, 100)
    result = swiglu(value, gate)
    print(f"swiglu(value, gate) shape: {result.shape}")
    print(f"Contains NaN: {torch.isnan(result).any().item()}")
    print(f"Contains Inf: {torch.isinf(result).any().item()}")
