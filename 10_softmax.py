"""
LLM Pre-training, Part 3 - Prediction with softmax.

The network outputs one raw score ("logit") per token in its vocabulary.
Logits can be negative and don't sum to 1. Softmax turns them into probabilities:
    p_i = e^(logit_i) / sum_j e^(logit_j)
Every p_i lands between 0 and 1, and they all add up to exactly 1 (100%).
"""

import math

# Logits after the input "The cat sat on the" (the video's simplified 3-token table).
logits = {"mat": 3.2, "rug": 1.3, "moon": -2.1}

# Step 1: exponentiate each logit. e^x is always positive, and bigger scores grow much faster.
exps = {tok: math.exp(z) for tok, z in logits.items()}
# Step 2: add up all the exponentiated scores.
total = sum(exps.values())
# Step 3: divide each one by the total, so they share 100% between them.
probs = {tok: e / total for tok, e in exps.items()}

print("Softmax, step by step (input: 'The cat sat on the ...')")
print(f"{'token':<6} | {'logit':>6} | {'e^logit':>8} | {'probability':>11}")
for tok in logits:
    print(f"{tok:<6} | {logits[tok]:>6.1f} | {exps[tok]:>8.2f} | {probs[tok]:>10.1%}")
print(f"{'sum':<6} | {'':>6} | {total:>8.2f} | {sum(probs.values()):>10.1%}\n")


def softmax(scores):
    """Reusable softmax over a list of logits."""
    m = max(scores)                             # subtract the max first: doesn't change the answer
    exps = [math.exp(s - m) for s in scores]    # (e^(a-m)/e^(b-m) = e^a/e^b) but avoids overflow
    total = sum(exps)                           # for huge logits like 1000, where e^1000 = inf
    return [e / total for e in exps]


# The video's first list also had 'floor'. Adding a 4th option takes probability from the others.
tokens = ["mat", "rug", "floor", "moon"]
p = softmax([3.2, 1.3, 0.5, -2.1])
print("With 'floor' (logit 0.5) added:")
for t, pi in zip(tokens, p):
    print(f"  {t:<6} {pi:6.1%}")
print(f"  sum    {sum(p):6.1%}")
print(f"\nNumerically safe even for huge logits: softmax([1000, 999]) = "
      f"{[round(x, 4) for x in softmax([1000, 999])]}\n")

# Same probabilities, two different jobs:
print("TRAINING  : compare the probabilities to the true next token -> loss -> learn (Part 4).")
print("GENERATION: sample ONE token from the probabilities to write new text (Part 5).")
