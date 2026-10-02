"""
Transformers, Part 11 - Autoregressive generation: the payoff.

The whole algorithm is a four-step loop:
  1. predict   feed the sequence in, keep only the LAST position's logits
  2. sample    softmax -> probabilities -> roll the dice (torch.multinomial)
  3. append    add the new token to the end of the sequence
  4. repeat    (cropping to the last block_size tokens - the model's short-term memory)

Two creative knobs:
  temperature   divide the logits before softmax: < 1 = safer, > 1 = more surprising
  top_k         keep only the k most likely tokens, so rare nonsense is never picked

Run 29_training_loss.py first: it trains the model this lesson uses.
"""

import sys
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from gpt2_min import GPT2, GPTConfig

checkpoint = Path(__file__).parent / "checkpoints" / "tiny_gpt.pt"
if not checkpoint.exists():
    sys.exit("No trained model yet - run 29_training_loss.py first.")

saved = torch.load(checkpoint)
model = GPT2(GPTConfig(**saved["config"]))
model.load_state_dict(saved["model"])
model.eval()                                # inference mode: dropout off
vocab = saved["vocab"]
to_local = {g: i for i, g in enumerate(vocab)}
enc = tiktoken.get_encoding("gpt2")


def encode(text):
    return torch.tensor([[to_local[g] for g in enc.encode(text)]])    # shape (1, T)


def decode(ids):
    return enc.decode([vocab[i] for i in ids])


# ---------------------------------------------------------------------------
# 1. The loop by hand, printing what the model is thinking at each step.
# ---------------------------------------------------------------------------
torch.manual_seed(0)
idx = encode("The dog")
print(f"GENERATION LOOP BY HAND, prompt {decode(idx[0].tolist())!r}")
with torch.no_grad():                       # we are predicting, not learning: skip gradients
    for step in range(4):
        idx_cond = idx[:, -model.config.block_size:]               # crop to the context window
        logits, _ = model(idx_cond)                                 # 1. predict ...
        logits = logits[:, -1, :]                                   # ... keep only the last guess
        probs = F.softmax(logits, dim=-1)                           # 2. sample
        next_id = torch.multinomial(probs, num_samples=1)
        top = torch.topk(probs[0], 3)
        guesses = ", ".join(f"{decode([i])!r} {p:.0%}" for p, i in zip(top.values, top.indices.tolist()))
        idx = torch.cat((idx, next_id), dim=1)                      # 3. append
        print(f"  step {step + 1}: top guesses {guesses:<40} picked {decode([next_id.item()])!r}")
print(f"  result: {decode(idx[0].tolist())!r}\n")

# ---------------------------------------------------------------------------
# 2. The temperature knob reshapes the probabilities before sampling.
# ---------------------------------------------------------------------------
with torch.no_grad():
    logits = model(encode("The cat sat on the"))[0][0, -1]
print("TEMPERATURE - next-token probabilities after 'The cat sat on the'")
ids = torch.topk(logits, 4).indices
words = [decode([i]) for i in ids.tolist()]
print(f"  {'temperature':<13}" + "".join(f"{w!r:>10}" for w in words))
for temperature in [0.5, 1.0, 2.0]:
    p = F.softmax(logits / temperature, dim=-1)[ids]
    print(f"  {temperature:<13}" + "".join(f"{v:>10.0%}" for v in p))
print("  low T sharpens the distribution (predictable), high T flattens it (creative)\n")

# ---------------------------------------------------------------------------
# 3. The built-in GPT2.generate does exactly the loop above, plus the two knobs.
# ---------------------------------------------------------------------------
print("model.generate(prompt 'The', 20 new tokens)")
for temperature, top_k in [(0.5, None), (1.0, None), (1.0, 3), (2.0, None)]:
    torch.manual_seed(1)
    out = model.generate(encode("The"), max_new_tokens=20, temperature=temperature, top_k=top_k)
    text = decode(out[0].tolist()).replace("\n", " ")
    print(f"  temperature {temperature}, top_k {top_k!s:<4}: {text!r}")

# ---------------------------------------------------------------------------
# 4. What lesson 13's bigram model could not do: use context from far back.
# ---------------------------------------------------------------------------
print("\nCONTEXT - the token before the answer is ' speak' both times; only the city differs")
for prompt in ["In Paris the people speak", "In Berlin the people speak"]:
    with torch.no_grad():
        p = F.softmax(model(encode(prompt))[0][0, -1], dim=-1)
    french, german = (p[to_local[enc.encode(w)[0]]] for w in [" French", " German"])
    print(f"  {prompt!r:<30} -> French {french:.0%}, German {german:.0%}")
print("  A bigram model sees only ' speak' and must guess the same both times.")
print("  Self-attention lets ' speak' look back four tokens to the city.")
