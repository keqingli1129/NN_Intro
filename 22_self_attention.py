"""
Transformers, Part 3 - Self-attention with real tensors, then packaged as an nn.Module.

  step 1  project      q, k, v = q_proj(x), k_proj(x), v_proj(x)     (B, T, C) each
  step 2  score        scores = q @ k^T                               (B, T, T)
  step 3  scale        scores / sqrt(d_k)                             keeps softmax from saturating
  step 4  normalize    softmax along the last dimension               each row sums to 1
  step 5  aggregate    out = weights @ v                              (B, T, C) - same shape as x
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Part 1: the raw tensor walkthrough. Sentence "a crane ate fish": B=1, T=4, C=2.
# ---------------------------------------------------------------------------
B, T, C = 1, 4, 2
words = ["a", "crane", "ate", "fish"]
x = torch.tensor([[[0.5, 0.5],              # pretend output of the embedding layers
                   [3.5, 3.5],
                   [4.5, 0.5],
                   [4.0, 1.0]]])

torch.manual_seed(42)                       # reproducible "learned" weights
q_proj = nn.Linear(C, C, bias=False)        # three learnable projections of the same x
k_proj = nn.Linear(C, C, bias=False)
v_proj = nn.Linear(C, C, bias=False)

q, k, v = q_proj(x), k_proj(x), v_proj(x)                   # step 1
scores = q @ k.transpose(-2, -1)                            # step 2: (1,4,2) @ (1,2,4) -> (1,4,4)
d_k = k.size(-1)
scaled_scores = scores / math.sqrt(d_k)                     # step 3
attention_weights = F.softmax(scaled_scores, dim=-1)        # step 4
output = attention_weights @ v                              # step 5: (1,4,4) @ (1,4,2) -> (1,4,2)

print("SHAPES (always track them)")
for name, t in [("x", x), ("q / k / v", q), ("k^T", k.transpose(-2, -1)), ("scores", scores),
                ("attention_weights", attention_weights), ("output", output)]:
    print(f"  {name:<18} {tuple(t.shape)}")

print("\nATTENTION WEIGHTS (row = who is looking, column = who they look at)")
print("        " + "".join(f"{w:>8}" for w in words))
for w, row in zip(words, attention_weights[0]):
    print(f"  {w:<6}" + "".join(f"{p:>8.2f}" for p in row) + f"   sum = {row.sum():.2f}")
print("  'crane' attends most to itself - a word is usually most related to itself.")
print(f"\noutput (context-aware vectors), shape {tuple(output.shape)}:")
print(output.detach())


# ---------------------------------------------------------------------------
# Part 2: the same five steps as a reusable module, with ONE fused projection layer.
# ---------------------------------------------------------------------------
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)   # Q, K and V in one big matmul

    def forward(self, x):
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)    # carve (B,T,3C) into 3 x (B,T,C)
        scores = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        return F.softmax(scores, dim=-1) @ v


attn = SingleHeadSelfAttention(C)
with torch.no_grad():                       # load the three separate weights into the fused layer
    attn.c_attn.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight]))

print("\nMODULE with fused c_attn")
print(f"  c_attn weight shape: {tuple(attn.c_attn.weight.shape)}  (3*C rows: Q, K, V stacked)")
print(f"  same output as the manual walkthrough: {torch.allclose(attn(x), output)}")
print("  Problem: 'a' can already see 'fish', a word from its future - fixed in lesson 23.")
