"""
PyTorch, Part 1 - torch.tensor, the only data structure you need.

A tensor is a multi-dimensional array (like a NumPy array) with two superpowers:
it can live on a GPU, and it can track gradients automatically (Part 2).

The 5-step recipe this whole series builds toward:
  1. forward pass   y_hat = model(x)
  2. loss           loss = loss_fn(y_hat, y)
  3. backward       loss.backward()
  4. update         optimizer.step()
  5. reset          optimizer.zero_grad()
"""

import torch  # installed with `uv add torch`

torch.manual_seed(0)                            # fix the random generator so every run prints the same numbers

# ---------------------------------------------------------------------------
# Pattern 1: direct creation from data.
# ---------------------------------------------------------------------------
data = [[1, 2, 3], [4, 5, 6]]                   # an ordinary Python list of lists
my_tensor = torch.tensor(data)                  # copy it into a tensor with the same layout
print("Pattern 1 - from a Python list:")
print(my_tensor, "\n")

# ---------------------------------------------------------------------------
# Pattern 2: creation from a desired shape (how model weights get initialised).
# ---------------------------------------------------------------------------
shape = (2, 3)                                  # 2 rows, 3 columns - we know the shape, not the values
ones = torch.ones(shape)                        # every entry 1.0
zeros = torch.zeros(shape)                      # every entry 0.0
rand = torch.randn(shape)                       # random numbers from a normal (bell-curve) distribution
print("Pattern 2 - from a shape:")
print("ones:\n", ones)
print("zeros:\n", zeros)
print("randn (a model's random starting point):\n", rand, "\n")

# ---------------------------------------------------------------------------
# Pattern 3: creation by mimicking another tensor.
# ---------------------------------------------------------------------------
template = torch.tensor([[1, 2], [3, 4]])       # an existing tensor (integers)
new = torch.randn_like(template, dtype=torch.float)   # same shape as template, random floats
print("Pattern 3 - mimic another tensor:")
print("template:\n", template)
print("randn_like (same shape, new values, float type):\n", new, "\n")

# ---------------------------------------------------------------------------
# The three attributes you'll check constantly: shape, dtype, device.
# ---------------------------------------------------------------------------
t = torch.randn(2, 3)
print("Attributes of a 2x3 random tensor:")
print("  shape :", t.shape)                     # torch.Size([2, 3]) - #1 debugging tool (shape mismatches!)
print("  dtype :", t.dtype)                     # torch.float32 - the default for learnable numbers
print("  device:", t.device)                    # cpu (or cuda:0 if it lived on a GPU)
print("  GPU available on this machine?", torch.cuda.is_available())
if torch.cuda.is_available():                   # only move to GPU if there is one
    print("  moved to GPU:", t.to("cuda").device)
print()

# ---------------------------------------------------------------------------
# Why float32? Learning = tiny nudges. Integers can't be nudged.
# ---------------------------------------------------------------------------
w_int = torch.tensor(3)                         # an integer tensor
w_float = torch.tensor(3.0)                     # a float tensor
print("Nudging a weight by 0.001:")
print("  float:", w_float + 0.001)              # 3.0010 - the nudge survives
print("  int  :", (w_int + 0.001).to(torch.int64), " <- forced back to a whole number, the nudge is lost")
labels = torch.tensor([0, 2, 1])                # class labels are categories, so integers are fine
print("  class labels can stay integers:", labels.dtype)
