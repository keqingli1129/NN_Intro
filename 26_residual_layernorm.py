"""
Transformers, Part 7 - The glue: residual connections and layer normalization.

  x = x + self.attn(self.ln_1(x))     <- the "+ x" is the residual connection (the express lane)
  x = x + self.mlp(self.ln_2(x))      <- ln_1 / ln_2 are layer norms (the stabilizers)

Residual: the sub-layer only has to learn a small ADJUSTMENT to x, and during backprop the
gradient flows straight through the "+" - so even very deep stacks keep learning.
LayerNorm: rescales each token's vector to mean 0, std 1, then lets the model pick its own
scale (gamma) and shift (beta) - so every layer gets inputs of a predictable size.
"""

import torch
from torch import nn

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. The residual connection is just element-wise addition. The shape never changes.
# ---------------------------------------------------------------------------
x_initial = torch.tensor([0.2, 0.1, 0.3, 0.4])
attention_output = torch.tensor([0.1, -0.1, 0.2, -0.3])      # the adjustment the sub-layer proposes
x_after = x_initial + attention_output
print("RESIDUAL CONNECTION  x = x + sublayer(x)")
print(f"  original x  {[round(v, 2) for v in x_initial.tolist()]}")
print(f"  adjustment  {[round(v, 2) for v in attention_output.tolist()]}")
print(f"  new x       {[round(v, 2) for v in x_after.tolist()]}   shape {tuple(x_after.shape)} unchanged\n")

# ---------------------------------------------------------------------------
# 2. Why it matters: the vanishing gradient ("a long game of telephone").
# The same 30-layer stack, with and without the "+ x" express lane.
# ---------------------------------------------------------------------------
n_layers, width = 30, 16
layers = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])


def first_layer_gradient(use_residual):
    """How big is the learning signal that makes it back to the very first layer?"""
    layers.zero_grad()
    h = torch.randn(8, width)
    for layer in layers:
        update = torch.tanh(layer(h))
        h = h + update if use_residual else update
    h.pow(2).mean().backward()              # any loss will do
    return layers[0].weight.grad.norm().item()


print(f"VANISHING GRADIENT - {n_layers} stacked layers, gradient size at the first layer:")
print(f"  without residuals  {first_layer_gradient(use_residual=False):.2e}   <- the signal has vanished")
print(f"  with residuals     {first_layer_gradient(use_residual=True):.2e}   <- still strong\n")

# ---------------------------------------------------------------------------
# 3. Layer normalization by hand, for one token with C = 4.
# ---------------------------------------------------------------------------
x_token = torch.tensor([0.3, -0.2, 0.8, 0.5])
mu = x_token.mean()
var = x_token.var(unbiased=False)           # LayerNorm divides by C, not C - 1
eps = 1e-5                                  # tiny number so we never divide by zero
x_hat = (x_token - mu) / torch.sqrt(var + eps)
print("LAYER NORM BY HAND")
print(f"  step 1 input      {[round(v, 2) for v in x_token.tolist()]}   mean {mu:.2f}, std {var.sqrt():.2f}")
print(f"  step 2 x_hat      {[round(v, 2) for v in x_hat.tolist()]}   "
      f"mean {x_hat.mean():.2f}, std {x_hat.std(unbiased=False):.2f}")
# (The video's numbers divide by a slightly different std, 0.41, so they differ a little.)

ln = nn.LayerNorm(4)
print(f"  nn.LayerNorm starts with gamma = {ln.weight.tolist()} and beta = {ln.bias.tolist()}")
print(f"  so at first it equals x_hat: {torch.allclose(ln(x_token), x_hat, atol=1e-6)}")

# Pretend training has learned a different scale and shift for the next layer.
with torch.no_grad():
    ln.weight.copy_(torch.tensor([1.5, 1.0, 1.0, 1.0]))     # gamma
    ln.bias.copy_(torch.tensor([0.5, 0.0, 0.0, 0.0]))       # beta
y = ln(x_token)
print(f"  step 3 y = gamma * x_hat + beta = {[round(v, 2) for v in y.tolist()]}")
print(f"         by hand:                   "
      f"{[round(v, 2) for v in (ln.weight * x_hat + ln.bias).tolist()]}\n")

# ---------------------------------------------------------------------------
# 4. Pre-norm (GPT-2) vs post-norm (the original 2017 transformer).
# ---------------------------------------------------------------------------
print(f"{'style':<11}{'equation':<30}{'used by':<24}{'training'}")
print(f"{'pre-norm':<11}{'x + sublayer(LayerNorm(x))':<30}{'GPT-2 (our code)':<24}"
      "more stable for deep stacks")
print(f"{'post-norm':<11}{'LayerNorm(x + sublayer(x))':<30}{'original transformer':<24}"
      "often needs learning-rate warm-up")
