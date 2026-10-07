"""
Transformers, Part 8 - The transformer block: the repeating Lego brick of GPT.

    class Block(nn.Module):
        def forward(self, x):
            x = x + self.attn(self.ln_1(x))    # the meeting: tokens communicate
            x = x + self.mlp(self.ln_2(x))     # desk time: each token thinks
            return x

Mantra: normalize, process, add. Output shape == input shape (B, T, C), so blocks stack.
Width = heads inside one block (12 specialists in one meeting).
Depth = blocks stacked (12 meetings, each refining the last).
"""

import torch
from torch import nn

from gpt2_min import Block, GPTConfig

torch.manual_seed(0)
config = GPTConfig(block_size=16, n_layer=4, n_head=4, n_embd=32)   # small enough to print
block = Block(config)
print(block, "\n")

# ---------------------------------------------------------------------------
# 1. The forward pass is exactly "normalize, process, add" - twice.
# ---------------------------------------------------------------------------
x = torch.randn(2, 10, config.n_embd)       # (B, T, C)
after_attention = x + block.attn(block.ln_1(x))                     # sub-layer 1
after_mlp = after_attention + block.mlp(block.ln_2(after_attention))  # sub-layer 2
print("ONE BLOCK")
print(f"  input  {tuple(x.shape)} -> output {tuple(block(x).shape)}   (B, T, C) in and out")
print(f"  block(x) == normalize/process/add by hand: {torch.allclose(block(x), after_mlp)}")
print(f"  parameters in one block: {sum(p.numel() for p in block.parameters()):,}\n")

# ---------------------------------------------------------------------------
# 2. Stacking: nn.ModuleList holds n_layer INDEPENDENT blocks.
# ---------------------------------------------------------------------------
h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
out = x
for i, blk in enumerate(h):
    out = blk(out)
    print(f"  after block {i}: {tuple(out.shape)}")

w0, w1 = h[0].attn.c_attn.weight, h[1].attn.c_attn.weight
print(f"\n  blocks share weights? {w0 is w1}  - each block has its own matrix "
      f"(block 0 starts {w0[0, 0]:.4f}, block 1 starts {w1[0, 0]:.4f})")
print("  Layer 1 sees raw embeddings, the last layer makes final refinements:")
print("  different jobs need different weights.")
print(f"  parameters in the {config.n_layer}-block stack: {sum(p.numel() for p in h.parameters()):,}"
      f" = {config.n_layer} x {sum(p.numel() for p in block.parameters()):,}\n")

print(f"{'concept':<8}{'comes from':<22}{'what it does':<38}{'analogy'}")
print(f"{'width':<8}{'multi-head attention':<22}{'parallel processing within a layer':<38}"
      "12 specialists in one meeting")
print(f"{'depth':<8}{'stacked blocks':<22}{'sequential processing across layers':<38}"
      "12 meetings, each refining the last")
