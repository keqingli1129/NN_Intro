"""
PyTorch, Part 5 - torch.nn: ready-made Lego bricks.

Instead of loose W and b tensors, layers package their parameters for you.
These exact bricks are inside GPT, Llama and Gemini.
"""

import torch
import torch.nn as nn

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# nn.Linear - does x @ W + b, but owns its W and b.
# ---------------------------------------------------------------------------
D_in, D_out = 1, 1
linear_layer = nn.Linear(in_features=D_in, out_features=D_out)
print("nn.Linear parameters (created for you, requires_grad=True already):")
print("  weight:", linear_layer.weight)
print("  bias  :", linear_layer.bias)
x = torch.tensor([[1.0], [2.0], [3.0]])         # 3 samples, 1 feature each
print("  linear_layer(x) =", linear_layer(x).detach().flatten())   # call it like a function = forward pass
# Note: PyTorch stores weight as (out, in) and computes x @ weight.T + bias.
manual = x @ linear_layer.weight.T + linear_layer.bias
print("  same as x @ W.T + b:", torch.allclose(manual, linear_layer(x)), "\n")

# Why activations: two stacked linear layers are still just ONE straight line.
l1, l2 = nn.Linear(1, 4), nn.Linear(4, 1)
W_combined = l2.weight @ l1.weight              # the two weight matrices multiply into one
b_combined = l2.weight @ l1.bias + l2.bias
print("Linear(Linear(x)) equals one Linear layer:",
      torch.allclose(l2(l1(x)), x @ W_combined.T + b_combined), " <- need non-linearity!\n")

# ---------------------------------------------------------------------------
# Activation functions - the "kinks" that let networks learn curves.
# ---------------------------------------------------------------------------
sample = torch.tensor([-2.0, -0.5, 0.5, 2.0])
relu = nn.ReLU()                                # max(0, x): negatives snapped to 0
gelu = nn.GELU()                                # smooth version used in GPT / Llama: negatives squashed
print("input:", sample)
print("ReLU :", relu(sample))
print("GELU :", gelu(sample), "\n")

# Softmax: logits -> probabilities, on the last dimension (each row separately).
softmax = nn.Softmax(dim=-1)
logits = torch.tensor([[1.0, 2.0, 3.0, 0.5],    # item 1: 4 class scores
                       [0.1, 0.2, 0.3, 4.0]])   # item 2
probs = softmax(logits)
print("Softmax probabilities:\n", probs)
print("row sums:", probs.sum(dim=-1), " (each row sums to 1; item 1's 3.0 -> highest probability)\n")

# ---------------------------------------------------------------------------
# The LLM bricks: Embedding, LayerNorm, Dropout.
# ---------------------------------------------------------------------------
vocab_size, embedding_dim = 10, 3               # 10 possible words, each becomes a 3-number vector
embedding_layer = nn.Embedding(vocab_size, embedding_dim)   # a learnable lookup table, shape 10x3
input_ids = torch.tensor([[1, 5, 0, 8]])        # one "sentence" of 4 token IDs, shape (1, 4)
word_vectors = embedding_layer(input_ids)       # look up row 1, row 5, row 0, row 8
print(f"Embedding: input {tuple(input_ids.shape)} -> output {tuple(word_vectors.shape)}")
print(word_vectors.detach())
print("  row for id 5 == embedding_layer.weight[5]:",
      torch.equal(word_vectors[0, 1], embedding_layer.weight[5]), "\n")

norm_layer = nn.LayerNorm(normalized_shape=3)   # normalise each 3-number vector
features = torch.tensor([[1.0, 2.0, 3.0], [100.0, 200.0, 400.0]])   # wildly different scales
normed = norm_layer(features)
print("LayerNorm:")
print("  output mean per row:", normed.mean(dim=-1).detach())           # ~0
print("  output std per row :", normed.std(dim=-1, unbiased=False).detach(), "\n")  # ~1

dropout_layer = nn.Dropout(p=0.5)               # zero out 50% of values during training
ones = torch.ones(1, 10)
dropout_layer.train()                           # training mode: dropout is ON
print("Dropout train():", dropout_layer(ones))  # random 0s, survivors scaled by 1/(1-0.5) = 2
dropout_layer.eval()                            # evaluation mode: dropout is OFF
print("Dropout eval() :", dropout_layer(ones))  # identity: all ones pass straight through
