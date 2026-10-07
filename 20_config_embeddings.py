"""
Transformers, Part 1 - The blueprint (GPTConfig) and the two embedding layers.

  token IDs  --wte-->  "what word am I?" vectors    \
                                                      +  ->  x, the input to the transformer blocks
  positions  --wpe-->  "where am I?" vectors        /

The same ~100 lines of gpt2_min.py build a toy model or full GPT-2: only the config changes.
"""

import torch
from torch import nn

from gpt2_min import GPTConfig

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. The blueprint: five knobs decide the model's size.
# ---------------------------------------------------------------------------
gpt2_small = GPTConfig()                    # the defaults are GPT-2 small
toy = GPTConfig(vocab_size=10, block_size=8, n_layer=2, n_head=1, n_embd=3)
print(f"{'knob':<12}{'meaning':<34}{'GPT-2 small':>12}{'toy':>6}")
for knob, meaning in [("vocab_size", "how many distinct tokens"),
                      ("block_size", "context window (tokens seen at once)"),
                      ("n_layer", "depth: blocks stacked"),
                      ("n_head", "attention heads per block"),
                      ("n_embd", "numbers per token vector")]:
    print(f"{knob:<12}{meaning:<34}{getattr(gpt2_small, knob):>12,}{getattr(toy, knob):>6}")

# ---------------------------------------------------------------------------
# 2. Why raw token IDs are useless to a network: the colour analogy.
# ---------------------------------------------------------------------------
print("\nBAD: colours as arbitrary IDs   red=1, orange=2, blue=8")
print("  |orange - red| = 1, |blue - orange| = 6 - but those distances mean nothing")
red, orange, blue = torch.tensor([0.9, 0.1]), torch.tensor([0.8, 0.2]), torch.tensor([0.1, 0.9])
print("GOOD: colours as vectors (redness, blueness)")
print(f"  distance red-orange = {torch.dist(red, orange):.2f}   (similar colours are close)")
print(f"  distance red-blue   = {torch.dist(red, blue):.2f}   (different colours are far apart)\n")

# ---------------------------------------------------------------------------
# 3. Token embeddings: nn.Embedding is just a learnable lookup table ("coordinate book").
# ---------------------------------------------------------------------------
token_embedding_table = nn.Embedding(toy.vocab_size, toy.n_embd)
print("TOKEN EMBEDDING TABLE (wte)")
print(f"  weight shape: {tuple(token_embedding_table.weight.shape)}  "
      "(one row of coordinates per token ID)")
print(f"  requires_grad: {token_embedding_table.weight.requires_grad}  "
      "<- training learns the coordinates")
print(token_embedding_table.weight.data)

idx = torch.tensor([5, 2, 1])               # three meaningless integer IDs ...
vectors = token_embedding_table(idx)        # ... become three 3-D vectors
print(f"\n  IDs {idx.tolist()} -> vectors of shape {tuple(vectors.shape)}")
print(f"  looking up ID 5 is just reading row 5: "
      f"{torch.equal(vectors[0], token_embedding_table.weight[5])}")

# The flaw: one ID always gives one vector, whatever the context.
BANK = 7                                    # pretend 7 is the ID of the word "bank"
river = token_embedding_table(torch.tensor([3, BANK]))[1]   # "river bank"
money = token_embedding_table(torch.tensor([9, BANK]))[1]   # "money ... bank"
print(f"  'bank' in two different sentences gives the same vector: {torch.equal(river, money)}")
print("  -> embeddings are context-free; fixing that is self-attention's job (lessons 21-24)\n")

# ---------------------------------------------------------------------------
# 4. Positional embeddings: one more lookup table, indexed by position instead of word.
# ---------------------------------------------------------------------------
B, T, C = 2, 5, toy.n_embd                  # batch, time (sequence length), channels
position_embedding_table = nn.Embedding(toy.block_size, C)  # 8 possible positions

idx = torch.tensor([[7, 3, 7, 1, 4],        # two sentences of 5 token IDs;
                    [2, 7, 5, 7, 0]])       # token 7 appears at several positions
tok_emb = token_embedding_table(idx)        # step 1: (B, T, C)
pos = torch.arange(0, T)                    # step 2: positions 0..T-1 only - T may be < block_size
pos_emb = position_embedding_table(pos)     #         (T, C)
x = tok_emb + pos_emb                       # step 3: broadcasting adds pos_emb to each sentence

print("POSITIONAL EMBEDDINGS (wpe)")
print(f"  positions used:  {pos.tolist()}  (block_size is {toy.block_size}, only {T} needed)")
print(f"  token embeddings     {tuple(tok_emb.shape)}")
print(f"  position embeddings  {tuple(pos_emb.shape)}   <- broadcast over the batch")
print(f"  combined x           {tuple(x.shape)}")
print(f"\n  token 7 at position 0, before: {[round(v, 4) for v in tok_emb[0, 0].tolist()]}")
print(f"  token 7 at position 2, before: {[round(v, 4) for v in tok_emb[0, 2].tolist()]}  (identical)")
print(f"  token 7 at position 0, after:  {[round(v, 4) for v in x[0, 0].tolist()]}")
print(f"  token 7 at position 2, after:  {[round(v, 4) for v in x[0, 2].tolist()]}  (now different)")
print("  Each token now knows WHAT it is and WHERE it is - but still not its context.")
