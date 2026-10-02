"""
Transformers, Part 5 - Multi-head attention: a team of specialists instead of one generalist.

One head has to track grammar, meaning and references all at once. Instead we split C into
n_head smaller heads that attend in parallel, then merge their results:

  split   (B, T, C) --view--> (B, T, nh, hs) --transpose--> (B, nh, T, hs)
  attend  each of the nh heads runs causal attention independently (nh acts like a batch dim)
  merge   (B, nh, T, hs) --transpose--> (B, T, nh, hs) --view--> (B, T, C) --c_proj--> (B, T, C)

GPT-2 small: C = 768, n_head = 12, so each head works with hs = 768 / 12 = 64 numbers.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from gpt2_min import CausalSelfAttention, GPTConfig

torch.manual_seed(0)
B, T, C, n_head = 1, 4, 768, 12
head_dim = C // n_head

# ---------------------------------------------------------------------------
# Step 1: split C into n_head heads of head_dim numbers each.
# ---------------------------------------------------------------------------
q = torch.randn(B, T, C)
q_reshaped = q.view(B, T, n_head, head_dim)       # carve up the last dimension
q_final = q_reshaped.transpose(1, 2)              # move heads to the front -> like a batch dim
print("STEP 1 - SPLIT")
print(f"  original q          {tuple(q.shape)!s:<16}(B, T, C)")
print(f"  after .view         {tuple(q_reshaped.shape)!s:<16}(B, T, nh, hs)")
print(f"  after .transpose    {tuple(q_final.shape)!s:<16}(B, nh, T, hs)\n")

# ---------------------------------------------------------------------------
# Step 2: attention runs 12 times at once - matmul only touches the last two dimensions.
# ---------------------------------------------------------------------------
k_final = torch.randn(B, T, C).view(B, T, n_head, head_dim).transpose(1, 2)
v_final = torch.randn(B, T, C).view(B, T, n_head, head_dim).transpose(1, 2)
att = q_final @ k_final.transpose(-2, -1) / math.sqrt(head_dim)   # (B,nh,T,hs) @ (B,nh,hs,T)
mask = torch.tril(torch.ones(T, T))
att = F.softmax(att.masked_fill(mask == 0, float("-inf")), dim=-1)
output_per_head = att @ v_final                                   # (B,nh,T,T) @ (B,nh,T,hs)
print("STEP 2 - ATTEND (all heads in parallel)")
print(f"  attention weights   {tuple(att.shape)!s:<16}(B, nh, T, T): one T x T map per head")
print(f"  output per head     {tuple(output_per_head.shape)!s:<16}(B, nh, T, hs)\n")

# ---------------------------------------------------------------------------
# Step 3: merge the heads back together, then mix them with a final projection.
# ---------------------------------------------------------------------------
merged = output_per_head.transpose(1, 2).contiguous().view(B, T, C)
c_proj = nn.Linear(C, C)
final_output = c_proj(merged)
print("STEP 3 - MERGE")
print(f"  merged              {tuple(merged.shape)!s:<16}(B, T, C)")
print(f"  after c_proj        {tuple(final_output.shape)!s:<16}(B, T, C) - same shape as x\n")
print("  (.contiguous() is needed because transpose only changes how memory is read;")
print("   .view needs the numbers laid out in order.)\n")

# ---------------------------------------------------------------------------
# The real module from gpt2_min.py does exactly these steps. Check it against a manual run.
# ---------------------------------------------------------------------------
config = GPTConfig(block_size=8, n_head=n_head, n_embd=C)
attn = CausalSelfAttention(config)
x = torch.randn(B, T, C)

q, k, v = attn.c_attn(x).split(C, dim=2)
q, k, v = (t.view(B, T, n_head, head_dim).transpose(1, 2) for t in (q, k, v))
att = (q @ k.transpose(-2, -1) / math.sqrt(head_dim)).masked_fill(mask == 0, float("-inf"))
y = (F.softmax(att, dim=-1) @ v).transpose(1, 2).contiguous().view(B, T, C)
manual = attn.c_proj(y)

print("gpt2_min.CausalSelfAttention")
print(f"  input {tuple(x.shape)} -> output {tuple(attn(x).shape)}")
print(f"  matches the manual split / attend / merge: {torch.allclose(attn(x), manual, atol=1e-6)}")
print(f"  parameters: {sum(p.numel() for p in attn.parameters()):,}  "
      "(c_attn 768x2304 + c_proj 768x768, plus biases)")
