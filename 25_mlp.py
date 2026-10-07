"""
Transformers, Part 6 - The MLP: the "thinking" layer.

Attention = communication (tokens talk to each other).
MLP       = computation   (each token thinks on its own, independently).

The MLP follows an "expand and contract" pattern:
  1. expand     c_fc:   C -> 4*C
  2. bend       GELU:   the non-linearity
  3. contract   c_proj: 4*C -> C
  4. dropout    random zeros during training only
"""

import torch
import torch.nn.functional as F
from torch import nn

from gpt2_min import MLP, GPTConfig

# ---------------------------------------------------------------------------
# 1. Demystify nn.Linear: output = input @ W^T + b. No magic.
# ---------------------------------------------------------------------------
linear_layer = nn.Linear(2, 4)              # C_in = 2 -> C_out = 4
with torch.no_grad():
    linear_layer.weight.copy_(torch.tensor([[1.0, 0.0],      # weight shape (C_out, C_in) = (4, 2)
                                            [-1.0, 0.0],
                                            [0.0, 2.0],
                                            [0.0, -2.0]]))
    linear_layer.bias.copy_(torch.tensor([1.0, 1.0, -1.0, -1.0]))   # bias shape (C_out,) = (4,)
input_vector = torch.tensor([0.5, -0.5])

W, b = linear_layer.weight, linear_layer.bias
by_hand = input_vector[0] * W[0, 0] + input_vector[1] * W[0, 1] + b[0]
print("nn.Linear BY HAND")
print(f"  output[0] = {input_vector[0]:.1f}*{W[0, 0]:.1f} + ({input_vector[1]:.1f})*{W[0, 1]:.1f}"
      f" + {b[0]:.1f} = {by_hand:.1f}")
print(f"  input  {input_vector.tolist()}")
print(f"  output {linear_layer(input_vector).tolist()}   (PyTorch agrees)")
print(f"  x @ W.T + b == layer(x): {torch.allclose(input_vector @ W.T + b, linear_layer(input_vector))}\n")

# ---------------------------------------------------------------------------
# 2. GELU: a smooth ReLU. Positives pass through almost unchanged, negatives get squashed.
# ---------------------------------------------------------------------------
xs = torch.tensor([-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
print("GELU vs ReLU")
print(f"  {'x':>6}{'ReLU':>8}{'GELU':>8}")
for x_val, r, g in zip(xs, F.relu(xs), F.gelu(xs)):
    print(f"  {x_val:>6.1f}{r:>8.2f}{g:>8.2f}")
print()

# ---------------------------------------------------------------------------
# 3. Trace one token (C = 2) through expand -> GELU -> contract -> dropout.
# ---------------------------------------------------------------------------
x = torch.tensor([[0.5, -0.5]])             # shape (1, 2)
torch.manual_seed(1)
fc = nn.Linear(2, 8)                        # expand: 2 -> 4*2 = 8
torch.manual_seed(2)
proj = nn.Linear(8, 2)                      # contract: 8 -> 2
dropout = nn.Dropout(0.5)

x_expanded = fc(x)
x_activated = F.gelu(x_expanded)
x_projected = proj(x_activated)
print("ONE TOKEN THROUGH THE MLP")
for name, t in [("input", x), ("1. after c_fc", x_expanded), ("2. after GELU", x_activated),
                ("3. after c_proj", x_projected)]:
    print(f"  {name:<16}{tuple(t.shape)!s:<9}{[round(v, 2) for v in t[0].tolist()]}")

dropout.eval()                              # inference: dropout does nothing
print(f"  4. dropout, eval mode:  {[round(v, 2) for v in dropout(x_projected)[0].tolist()]}  (unchanged)")
dropout.train()                             # training: zero random elements, scale the rest by 2
torch.manual_seed(0)
print(f"     dropout(p=0.5) on 8 ones, train mode: {dropout(torch.ones(8)).tolist()}")
print("     (about half become 0, survivors are scaled x2 so the average stays the same)\n")

# ---------------------------------------------------------------------------
# 4. The MLP module from gpt2_min.py: shape in == shape out, so it can be added back to x.
# ---------------------------------------------------------------------------
config = GPTConfig(n_embd=768)
mlp = MLP(config)
tokens = torch.randn(2, 5, 768)             # (B, T, C)
print("gpt2_min.MLP (GPT-2 small sizes)")
print(f"  {tuple(tokens.shape)} -> {tuple(mlp(tokens).shape)}   (B, T, C) in and out")
print(f"  hidden size: {mlp.c_fc.out_features:,} = 4 x 768,  parameters: "
      f"{sum(p.numel() for p in mlp.parameters()):,}")
# "Independently": changing token 0 leaves every other token's output alone.
changed = tokens.clone()
changed[0, 0] += 10.0
print(f"  token 1's output unchanged after editing token 0: "
      f"{torch.allclose(mlp(tokens)[0, 1], mlp(changed)[0, 1])}  (each token thinks alone)")
