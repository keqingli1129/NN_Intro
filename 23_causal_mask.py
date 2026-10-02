"""
Transformers, Part 4 - The causal mask: no peeking at the future.

GPT writes one token at a time, so the token at position t may only look at positions 0..t.
Fix: before softmax, set every "future" score to -inf. Since e^(-inf) = 0, softmax gives
those positions exactly 0 weight.

  mask   = torch.tril(torch.ones(T, T))                    1 = keep, 0 = future (block it)
  scores = scores.masked_fill(mask == 0, float("-inf"))
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# The same setup as lesson 22, so we pick up from its scaled scores.
B, T, C = 1, 4, 2
words = ["a", "crane", "ate", "fish"]
x = torch.tensor([[[0.5, 0.5], [3.5, 3.5], [4.5, 0.5], [4.0, 1.0]]])
torch.manual_seed(42)
q_proj, k_proj, v_proj = (nn.Linear(C, C, bias=False) for _ in range(3))
q, k, v = q_proj(x), k_proj(x), v_proj(x)
scaled_scores = (q @ k.transpose(-2, -1) / math.sqrt(C)).detach()


def show(title, matrix, fmt):
    print(title)
    print("        " + "".join(f"{w:>8}" for w in words))
    for w, row in zip(words, matrix):
        print(f"  {w:<6}" + "".join(f"{val:>8{fmt}}" for val in row.tolist()))
    print()


unmasked = F.softmax(scaled_scores, dim=-1)[0]
show("UNMASKED attention weights - every row peeks at the future:", unmasked, ".2f")

# Step 1: the mask. Row t has ones in columns 0..t only.
mask = torch.tril(torch.ones(T, T))
show("Step 1 - mask = torch.tril(torch.ones(T, T)):", mask, ".0f")

# Step 2: replace the future scores with -inf.
masked_scores = scaled_scores.masked_fill(mask == 0, float("-inf"))
show("Step 2 - scores after masked_fill (upper-right triangle is -inf):", masked_scores[0], ".2f")

# Step 3: re-run softmax. e^(-inf) = 0, so the future gets exactly zero weight.
causal = F.softmax(masked_scores, dim=-1)[0]
show("Step 3 - CAUSAL attention weights:", causal, ".2f")

print(f"{'question':<28}{'unmasked':>10}{'causal':>10}")
for looker, target in [("crane", "fish"), ("ate", "fish")]:
    i, j = words.index(looker), words.index(target)
    print(f"{f'can {looker!r} see {target!r}?':<28}{unmasked[i, j]:>10.0%}{causal[i, j]:>10.0%}")


# ---------------------------------------------------------------------------
# Packaged as a module. The mask is stored with register_buffer: it is part of the model's
# state (saved with it, moved to the GPU with it) but it is NOT a trainable parameter.
# ---------------------------------------------------------------------------
class CausalSingleHeadAttention(nn.Module):
    def __init__(self, n_embd, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        T = x.size(1)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        scaled_scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        # The one new line: slice the stored mask to the current length T, then hide the future.
        scaled_scores = scaled_scores.masked_fill(self.bias[:T, :T] == 0, float("-inf"))
        return F.softmax(scaled_scores, dim=-1) @ v


attn = CausalSingleHeadAttention(n_embd=C, block_size=8)
print("\nMODULE")
print(f"  trainable parameters: {[name for name, _ in attn.named_parameters()]}")
print(f"  buffers:              {[name for name, _ in attn.named_buffers()]}")
print(f"  saved in state_dict:  {list(attn.state_dict())}")

# Proof that it cannot cheat: change the LAST word and only the last output changes.
x_changed = x.clone()
x_changed[0, -1] = torch.tensor([-5.0, 9.0])               # replace "fish" with something else
out, out_changed = attn(x), attn(x_changed)
for t, w in enumerate(words):
    print(f"  output for {w!r:<8} unchanged after editing 'fish': "
          f"{torch.allclose(out[0, t], out_changed[0, t])}")
