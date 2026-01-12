# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hands-on learning project for mastering Large Language Models through practical implementation. The codebase contains educational implementations of core LLM components (attention mechanisms, position encoding, MoE), reinforcement learning examples, and tensor manipulation exercises.

## Repository Structure

```
learn-llm-by-hands/
├── NLP/                    # Core LLM component implementations
│   ├── GQA.py             # Grouped Query Attention with RoPE
│   ├── MLA.py             # Multi-Head Latent Attention
│   ├── Moe.py             # Mixture of Experts (DeepSeek-V3 style)
│   ├── flash-attn.py      # Flash Attention 2
│   ├── RMSNorm.py         # Root Mean Square Layer Normalization
│   ├── rotary embedding.ipynb  # RoPE variants (NTK, YaRN, Linear scaling)
│   └── *.md               # Mathematical explanations for each component
├── RL/                     # Reinforcement Learning examples
│   ├── frozenlake_policy_iteration.py
│   ├── frozenlake_value_iteration.py
│   └── PONG game.py       # Deep Q-Learning for Atari Pong
└── tensor puzzles/         # Vectorized tensor exercises
    ├── puzzles*.py        # Individual tensor manipulation puzzles
    └── utils.py           # Helper functions (arange, where)
```

## Running Code

**Python files**: Most files are standalone and can be run directly:
```bash
python NLP/GQA.py
python RL/frozenlake_value_iteration.py
```

**Jupyter notebooks**: Use Jupyter or VS Code with Jupyter extension:
```bash
jupyter notebook "NLP/rotary embedding.ipynb"
```

**Tensor puzzles**: Each puzzle file demonstrates a specific tensor operation without for-loops:
```bash
python "tensor puzzles/puzzles1 ones.py"
```

## Key Implementation Patterns

### Attention Mechanisms

- **RoPE Implementation** (`GQA.py`): Uses complex number multiplication for efficient rotary position embedding. The trick is `torch.polar(torch.ones_like(freqs), freqs)` to create rotation factors, then `torch.view_as_complex`/`torch.view_as_real` for application.

- **GQA (Grouped Query Attention)**: Key-value heads are fewer than query heads. Uses `repeak_kv()` function with `expand()` and `reshape()` to create views (not copies) for repeating KV tensors across multiple query heads.

- **KV Cache**: Attention modules pre-allocate cache tensors of shape `[max_batch_size, max_seq_len, n_kv_heads, head_dim]` and update them incrementally during inference.

### Mixture of Experts

- **Token-level routing** (`Moe.py`): MoE reshapes input from `[batch, seq_len, dim]` to `[batch * seq_len, dim]` for token-wise expert selection
- **Expert execution**: Loop through experts, use `torch.where(indices == i)` to find tokens routed to each expert
- **Shared experts**: Combine with routed expert outputs: `(y + z).view(shape)`

### Tensor Puzzles

- Use `arange(i)` from `utils.py` instead of `range()` or `torch.arange()`
- Use `where(condition, a, b)` instead of if-statements
- Goal: Implement operations using only vectorized operations, no loops

## Dependencies

The project uses primarily:
- **PyTorch** - All deep learning implementations
- **Gymnasium** + **ALE** - For RL environments (Pong)
- **NumPy** - Numerical operations

No unified `requirements.txt` exists; dependencies are minimal and per-file.

## Code Conventions

- **Comments**: Mixed English and Chinese; Chinese comments explain mathematical concepts
- **Documentation**: Each major implementation has a companion `.md` file with mathematical derivations
- **Type hints**: Used selectively in complex modules (GQA, Moe)
- **Dataclasses**: `ModelArgs` dataclass for configuration (similar to Llama-style config)

## PR Agent Configuration

The repository uses PR Agent with MiniMax models (configured in `.pr_agent.toml`):
```toml
model="MiniMax-M2.1"
fallback_models=["MiniMax-M2.1-lightning", "MiniMax-M2"]
```

Triggered automatically on PR events via GitHub Actions.
