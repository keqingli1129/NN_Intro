"""
LLM Pre-training, Part 4 - Learning with cross-entropy loss ("surprise").

    loss = -log(probability the model gave to the CORRECT token)

Confident and right  -> small surprise -> small loss.
Unsure (or wrong)    -> big surprise   -> big loss.
Every other token's probability is ignored in the formula (but softmax ties them together).
"""

import math


def softmax(scores):
    m = max(scores)                             # stability trick from Part 3
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def cross_entropy(p_correct):
    """Surprise: negative natural log of the probability assigned to the right answer."""
    return -math.log(p_correct)


# The two scenarios from the video.
print("Correct next token: 'mat'")
print(f"  good prediction, p(mat) = 86.6% -> loss = -ln(0.866) = {cross_entropy(0.866):.2f}  (low surprise)")
print(f"  bad prediction,  p(mat) =  1.0% -> loss = -ln(0.01)  = {cross_entropy(0.01):.2f}  (high surprise)\n")

# How the loss grows as confidence in the right answer falls.
print(f"{'p(correct)':>10} | {'loss':>6}")
for p in [0.99, 0.9, 0.5, 0.25, 0.1, 0.01, 0.001]:
    print(f"{p:>10.1%} | {cross_entropy(p):>6.2f}")
print("p = 100% gives loss 0; p -> 0 makes the loss shoot toward infinity.")
print(f"A model guessing uniformly over GPT-2's 50,257 tokens: loss = ln(50257) = {math.log(50257):.2f}\n")

# ---------------------------------------------------------------------------
# From loss to learning: the backprop starting point.
# For softmax + cross-entropy the gradient of the loss w.r.t. each logit is beautifully simple:
#     dLoss/dlogit_i = p_i - (1 if i is the correct token else 0)
# Correct token: p - 1 (negative) -> gradient descent RAISES its logit.
# Wrong tokens : p     (positive) -> gradient descent LOWERS their logits.
# ---------------------------------------------------------------------------
tokens = ["mat", "rug", "moon"]
logits = [3.2, 1.3, -2.1]
correct = 0                                     # index of 'mat'

probs = softmax(logits)
grads = [p - (1 if i == correct else 0) for i, p in enumerate(probs)]

# Check the formula numerically, exactly as we did for the chain rule in lesson 03.
h = 1e-6
numeric = []
for i in range(len(logits)):
    up = logits.copy(); up[i] += h               # nudge logit i up a little
    down = logits.copy(); down[i] -= h           # ... and down a little
    numeric.append((cross_entropy(softmax(up)[correct]) - cross_entropy(softmax(down)[correct])) / (2 * h))

print("Gradient of the loss w.r.t. each logit (p - one_hot):")
for t, g, n in zip(tokens, grads, numeric):
    print(f"  {t:<5} formula {g:+.4f}   numerical {n:+.4f}")

# Pretend the logits themselves are the knobs and take a few gradient-descent steps.
# In a real LLM these gradients keep flowing backward into billions of weights.
lr = 1.0
print("\nGradient descent on the logits (target 'mat'):")
for step in range(4):
    probs = softmax(logits)
    print(f"  step {step}: p(mat) = {probs[correct]:.1%}, loss = {cross_entropy(probs[correct]):.4f}")
    grads = [p - (1 if i == correct else 0) for i, p in enumerate(probs)]
    logits = [z - lr * g for z, g in zip(logits, grads)]
