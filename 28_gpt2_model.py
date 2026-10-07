"""
Transformers, Part 9 - The full GPT-2: embeddings -> blocks -> final LayerNorm -> LM head.

  token IDs (B, T)
    -> wte(idx) + wpe(pos)           (B, T, C)            lesson 20
    -> dropout -> n_layer x Block    (B, T, C)            lessons 21-27
    -> ln_f                          (B, T, C)            lesson 26
    -> lm_head                       (B, T, vocab_size)   this lesson: a score for every token

Two ideas are new here:
  - the LM head makes T predictions at once, one per position (all used in training,
    only the last one used when generating);
  - weight tying: lm_head.weight IS wte.weight - one matrix, used in both directions.
"""

import torch

from gpt2_min import GPT2, GPTConfig

torch.manual_seed(0)
config = GPTConfig(vocab_size=50, block_size=8, n_layer=2, n_head=2, n_embd=16)
model = GPT2(config)
model.eval()                                # no dropout while we inspect it

# ---------------------------------------------------------------------------
# 1. The forward pass, one stage at a time (the same code as GPT2.forward).
# ---------------------------------------------------------------------------
idx = torch.tensor([[5, 12, 8, 21]])        # "a crane ate fish" as made-up token IDs: B=1, T=4
T = idx.size(1)
pos = torch.arange(0, T)
x = model.drop(model.wte(idx) + model.wpe(pos))
print("FORWARD PASS")
print(f"  token IDs                {tuple(idx.shape)}")
print(f"  embeddings               {tuple(x.shape)}")
for i, block in enumerate(model.h):
    x = block(x)
    print(f"  after block {i}            {tuple(x.shape)}")
x = model.ln_f(x)
print(f"  after ln_f               {tuple(x.shape)}")
logits = model.lm_head(x)
print(f"  logits (lm_head)         {tuple(logits.shape)}   (B, T, vocab_size)")
print(f"  same as model(idx): {torch.allclose(logits, model(idx)[0])}\n")

# ---------------------------------------------------------------------------
# 2. T predictions at once. Thanks to the causal mask, position t only saw tokens 0..t.
# ---------------------------------------------------------------------------
words = ["a", "crane", "ate", "fish"]
print("ONE PREDICTION PER POSITION")
for t in range(T):
    context = " ".join(words[: t + 1])
    target = f"compare with {words[t + 1]!r}" if t + 1 < T else "used when generating"
    print(f"  logits[0, {t}] guesses what follows {context!r:<20} -> {target}")
print("  Training uses all T guesses at once; generation keeps only logits[:, -1, :].\n")

# ---------------------------------------------------------------------------
# 3. Weight tying: the LM head and the token embedding are the same matrix in memory.
# ---------------------------------------------------------------------------
print("WEIGHT TYING")
print(f"  wte.weight      {tuple(model.wte.weight.shape)}  ID -> meaning  (row lookup)")
print(f"  lm_head.weight  {tuple(model.lm_head.weight.shape)}  meaning -> ID  (score every row)")
print(f"  the very same tensor object: {model.lm_head.weight is model.wte.weight}")
with torch.no_grad():
    model.wte.weight[0, 0] = 123.0          # change one number through the embedding ...
print(f"  set wte.weight[0, 0] = 123 -> lm_head.weight[0, 0] = {model.lm_head.weight[0, 0]:.0f}\n")

# ---------------------------------------------------------------------------
# 4. Real GPT-2 small sizes. Built on the "meta" device: shapes only, no memory used.
# ---------------------------------------------------------------------------
with torch.device("meta"):
    gpt2_small = GPT2(GPTConfig())
total = sum(p.numel() for p in gpt2_small.parameters())    # parameters() counts a shared tensor once
tied = gpt2_small.wte.weight.numel()
print("GPT-2 SMALL (124M)")
print(f"  total parameters:        {total:>12,}")
print(f"  embedding matrix:        {tied:>12,}  = 50,257 x 768")
print(f"  without weight tying:    {total + tied:>12,}  (a second copy for the LM head)")
print(f"  saved by one line of code: {tied / 1e6:.1f} million parameters")
