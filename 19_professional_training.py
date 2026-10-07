"""
PyTorch, Part 6 - The professional way: nn.Module + torch.optim, and the link to LLMs.

The from-scratch loop in lesson 17 becomes:
    y_hat = model(X)                  # 1. forward
    loss = loss_fn(y_hat, y_true)     # 2. loss
    optimizer.zero_grad()             # \
    loss.backward()                   #  > the three-line mantra (steps 5, 3, 4)
    optimizer.step()                  # /
"""

import torch
from torch import nn, optim

torch.manual_seed(42)

# Same data as lesson 17: y = 2x + 1 + noise.
N, D_in, D_out = 10, 1, 1
X = torch.randn(N, D_in)
y_true = X @ torch.tensor([[2.0]]) + 1.0 + 0.1 * torch.randn(N, D_out)


# ---------------------------------------------------------------------------
# The model blueprint: inherit nn.Module, define layers in __init__, connect them in forward.
# ---------------------------------------------------------------------------
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()                      # REQUIRED: sets up nn.Module's parameter tracking
        self.linear_layer = nn.Linear(in_features, out_features)   # registered automatically

    def forward(self, x):
        return self.linear_layer(x)             # how data flows through the layers


model = LinearRegressionModel(in_features=1, out_features=1)
print(model)                                    # prints the architecture
print("parameters:", [tuple(p.shape) for p in model.parameters()], "\n")  # found W and b for us

# ---------------------------------------------------------------------------
# The optimizer (replaces the manual update + zeroing) and a ready-made loss.
# ---------------------------------------------------------------------------
learning_rate = 0.1                             # the video uses 0.01; with only 100 epochs Adam then
                                                # moves W by at most ~1 in total, so we use 0.1 here
optimizer = optim.Adam(model.parameters(), lr=learning_rate)   # Adam: the default first choice
loss_fn = nn.MSELoss()                          # the same mean squared error we wrote by hand

# ---------------------------------------------------------------------------
# The clean training loop.
# ---------------------------------------------------------------------------
for epoch in range(100):
    y_hat = model(X)                            # 1. forward pass
    loss = loss_fn(y_hat, y_true)               # 2. loss
    optimizer.zero_grad()                       # 5. reset old gradients (done first, before backward)
    loss.backward()                             # 3. compute new gradients
    optimizer.step()                            # 4. update every parameter the optimizer manages
    if epoch % 10 == 0:
        print(f"epoch {epoch:>3}: loss = {loss.item():.4f}")

W, b = model.linear_layer.weight.item(), model.linear_layer.bias.item()
print(f"\nLearned: W = {W:.4f}, b = {b:.4f}   (truth: W = 2, b = 1)\n")


# ---------------------------------------------------------------------------
# The link to LLMs: the feed-forward network inside every transformer block.
# ---------------------------------------------------------------------------
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_hidden)   # expand
        self.activation = nn.GELU()                  # non-linearity
        self.layer2 = nn.Linear(d_hidden, d_model)   # project back

    def forward(self, x):
        return self.layer2(self.activation(self.layer1(x)))


ffn = FeedForwardNetwork(d_model=8, d_hidden=32)     # toy size so it runs instantly
tokens = torch.randn(1, 5, 8)                        # 1 sentence, 5 tokens, 8 numbers per token
print(ffn)
print(f"FFN: input {tuple(tokens.shape)} -> output {tuple(ffn(tokens).shape)}\n")

# Llama 3 8B sizes. Only count the parameters - actually building it would need ~470 MB of RAM.
d_model, d_hidden = 4096, 14336
print(f"One Llama-3-8B FFN weight matrix: {d_model:,} x {d_hidden:,} = {d_model * d_hidden:,} numbers")
print(f"{'':<16}{'our toy model':<20}{'Llama 3 8B'}")
print(f"{'model':<16}{'nn.Linear':<20}{'transformer (32 blocks)'}")
print(f"{'layer':<16}{'nn.Linear':<20}{'nn.Linear (inside FFN & attention)'}")
print(f"{'W shape':<16}{'1 x 1':<20}{'4096 x 14336'}")
print(f"{'parameters':<16}{sum(p.numel() for p in model.parameters()):<20}{'~8,000,000,000'}")
print(f"{'loss':<16}{'MSE':<20}{'cross-entropy (next token)'}")
print(f"{'training step':<16}{'zero_grad / backward / step  - identical'}")
