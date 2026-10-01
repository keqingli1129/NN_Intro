"""
LLM Pre-training, Part 5 - Controlled generation: temperature and top-p.

When generating, the model must pick ONE token. Always picking the top one (greedy) is
boring and loops, so we SAMPLE - and steer the sampling with two knobs:
  temperature : divide logits by T before softmax. T < 1 sharpens, T > 1 flattens.
  top-p       : keep only the smallest set of top tokens whose probabilities add up to p,
                then sample from that "nucleus". Cuts off the junk in the long tail.
"""

import math
import random
from collections import Counter


def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def apply_temperature(logits, temperature):
    """Adjusted logits = logits / T  (the 'creativity knob')."""
    return [z / temperature for z in logits]


def top_p_filter(tokens, probs, top_p):
    """Nucleus sampling: keep the most likely tokens until their total reaches top_p."""
    ranked = sorted(zip(tokens, probs), key=lambda tp: tp[1], reverse=True)  # 1) sort high -> low
    nucleus, running = [], 0.0
    for tok, p in ranked:                       # 2) add them up one by one ...
        nucleus.append((tok, p))
        running += p
        if running >= top_p:                    # ... until we hit the threshold
            break
    total = sum(p for _, p in nucleus)          # 3) renormalise so the nucleus sums to 1
    return [(tok, p / total) for tok, p in nucleus]


def sample_next(tokens, logits, temperature=1.0, top_p=1.0, rng=random):
    """The full generation step: temperature -> softmax -> top-p -> random draw."""
    probs = softmax(apply_temperature(logits, temperature))
    nucleus = top_p_filter(tokens, probs, top_p)
    choices, weights = zip(*nucleus)
    return rng.choices(choices, weights=weights)[0]   # pick one token, weighted by probability


# Prompt: "The weather today is ..."
# The video shows probabilities, not logits. Taking log(p) gives logits that reproduce them at T=1.
# (The video's exact table can't be recovered, so the T=0.5 / T=2 numbers differ slightly from it.)
tokens = ["sunny", "cloudy", "rainy", "beautiful", "cold"]
base_probs = [0.40, 0.30, 0.15, 0.10, 0.05]
logits = [math.log(p) for p in base_probs]

print("Prompt: 'The weather today is ...'\n")
temps = [0.2, 0.5, 1.0, 1.5, 2.0]
print(f"{'token':<10}" + "".join(f"{'T=' + str(t):>9}" for t in temps))
table = {t: softmax(apply_temperature(logits, t)) for t in temps}
for i, tok in enumerate(tokens):
    print(f"{tok:<10}" + "".join(f"{table[t][i]:>9.1%}" for t in temps))
print("Low T -> 'sunny' dominates (precise, factual). High T -> flatter (creative, 'beautiful' gains).\n")

for p in [0.1, 0.5, 0.9]:
    nucleus = top_p_filter(tokens, base_probs, p)
    print(f"top_p = {p}: sample only from {[t for t, _ in nucleus]}")
print()

# Sample 1,000 times with different settings and count what comes out.
rng = random.Random(42)                         # fixed seed so the results are reproducible
settings = [("greedy (always top)", None, None), ("T=0.5, top_p=1.0", 0.5, 1.0),
            ("T=1.0, top_p=0.9", 1.0, 0.9), ("T=1.5, top_p=1.0", 1.5, 1.0)]
for label, temp, top_p in settings:
    if temp is None:
        picks = [tokens[logits.index(max(logits))]] * 1000          # greedy: argmax, no randomness
    else:
        picks = [sample_next(tokens, logits, temp, top_p, rng) for _ in range(1000)]
    counts = Counter(picks)
    print(f"{label:<20}: " + ", ".join(f"{t} {counts[t]}" for t in tokens if counts[t]))
