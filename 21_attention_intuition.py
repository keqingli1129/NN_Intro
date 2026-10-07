"""
Transformers, Part 2 - Self-attention intuition: how "crane" figures out what it means.

  "The crane ate a fish."       -> crane is a bird
  "The crane lifted the steel." -> crane is a machine
Its starting vector is identical in both. Attention updates it from its neighbours:

  Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

  Q (query) - what this word is looking for
  K (key)   - what each word is (its label)
  V (value) - what each word offers (its "elevator pitch", a learned transform of x)

Three steps: 1. score (query . key)  2. normalize (softmax)  3. aggregate (weighted sum of values)
Like the video we use hand-picked 2-D vectors (dim 1 = "is an animal", dim 2 = "is a machine"),
only the words crane attends to, and skip the sqrt(d_k) scaling until lesson 22.
"""

import torch
import torch.nn.functional as F

crane_query = torch.tensor([0.7, 0.7])      # ambiguous: a bit of both
crane_key = torch.tensor([0.7, 0.7])
crane_value = torch.tensor([0.5, 0.5])

sentences = {
    "The crane ate a fish": {"ate": [0.9, 0.1], "fish": [0.8, 0.2]},          # animal words
    "The crane lifted the steel": {"lifted": [0.1, 0.9], "steel": [0.2, 0.8]},  # machine words
}

for sentence, context in sentences.items():
    words = ["crane", *context]
    # In this toy example each context word's key and value are the same vector.
    keys = torch.stack([crane_key] + [torch.tensor(v) for v in context.values()])
    values = torch.stack([crane_value] + [torch.tensor(v) for v in context.values()])

    scores = keys @ crane_query             # 1. SCORE: dot product of crane's query with every key
    weights = F.softmax(scores, dim=0)      # 2. NORMALIZE: scores -> percentages that sum to 1
    new_crane = weights @ values            # 3. AGGREGATE: weighted average of the values

    print(f"{sentence!r}")
    print(f"  {'word':<8}{'key':<14}{'score':>6}{'weight':>8}")
    for w, k, s, a in zip(words, keys, scores, weights):
        print(f"  {w:<8}{[round(x, 1) for x in k.tolist()]!s:<14}{s:>6.2f}{a:>8.0%}")
    print(f"  new crane = {' + '.join(f'{a:.2f}*{w}' for w, a in zip(words, weights))}")
    print(f"            = [{new_crane[0]:.2f}, {new_crane[1]:.2f}]   "
          f"(animal {new_crane[0]:.2f}, machine {new_crane[1]:.2f})\n")

print("The scores and weights are identical in both sentences - but they are applied to")
print("different values, so the same starting crane vector becomes two context-aware vectors.")
