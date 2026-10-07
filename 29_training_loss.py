"""
Transformers, Part 10 - Training: one number (the loss) teaches the whole network.

    logits, loss = model(idx, targets)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

Targets are the inputs shifted one position left (next-token prediction, lesson 08).
One training step, in six stages:
  1. forward pass       idx (B, T) -> logits (B, T, V)
  2. reshape            logits -> (B*T, V), targets -> (B*T,)
  3. cross-entropy      one loss per prediction: B*T of them
  4. average            their MEAN is the single scalar loss
  5. backward           loss.backward() - gradients for every weight
  6. update             optimizer.step()

Finally we train a tiny GPT-2 on lesson 13's corpus and save it for lesson 30.
"""

from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

from gpt2_min import GPT2, GPTConfig

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# 1. From a raw stream of token IDs to (input, target) pairs.
# ---------------------------------------------------------------------------
stream = torch.tensor([5, 12, 8, 21, 6, 33, 9, 4, 15, 7, 2])
T = 4                                       # block size
idx = torch.stack([stream[0:T], stream[T:2 * T]])               # B = 2 samples
targets = torch.stack([stream[1:T + 1], stream[T + 1:2 * T + 1]])  # shifted one to the left
print("INPUT / TARGET PAIRS from the stream", stream.tolist())
for b in range(2):
    print(f"  sample {b + 1}: idx {idx[b].tolist()} -> targets {targets[b].tolist()}")
print(f"  batch: idx {tuple(idx.shape)}, targets {tuple(targets.shape)}\n")

# ---------------------------------------------------------------------------
# 2. The six stages of one training step, on an untrained toy model.
# ---------------------------------------------------------------------------
config = GPTConfig(vocab_size=40, block_size=T, n_layer=2, n_head=2, n_embd=16)
model = GPT2(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

logits, loss = model(idx, targets)                                  # stage 1
flat_logits = logits.view(-1, logits.size(-1))                      # stage 2
flat_targets = targets.view(-1)
per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")   # stage 3
print("ONE TRAINING STEP")
print(f"  1. forward:   logits {tuple(logits.shape)}  (B, T, vocab_size)")
print(f"  2. reshape:   logits {tuple(flat_logits.shape)}, targets {tuple(flat_targets.shape)} "
      f"-> {flat_targets.tolist()}")
print(f"  3. per-token losses: {[round(v, 2) for v in per_token.tolist()]}")
print(f"  4. their mean: {per_token.mean():.4f}  == the model's loss {loss:.4f}  "
      f"(a mean, not a sum, so batch size doesn't change the scale)")
print(f"     untrained loss is about ln(vocab_size) = ln({config.vocab_size}) = "
      f"{torch.log(torch.tensor(float(config.vocab_size))):.4f} - a uniform guess")
optimizer.zero_grad()
loss.backward()                                                     # stage 5
optimizer.step()                                                    # stage 6
print(f"  5. backward:  {sum(p.grad is not None for p in model.parameters())} weight tensors "
      "got gradients from that ONE number")
print(f"  6. update:    loss on the same batch is now {model(idx, targets)[1]:.4f}\n")

# ---------------------------------------------------------------------------
# 3. Train a tiny GPT-2 for real, on the same small corpus as lesson 13.
# ---------------------------------------------------------------------------
corpus = """The cat sat on the mat. The dog sat on the rug. The cat chased the dog.
The dog chased the ball. The cat slept on the mat. The dog slept on the rug.
In Paris the people speak French. In Berlin the people speak German.
The weather today is sunny. The weather today is cloudy. The weather today is cold."""

enc = tiktoken.get_encoding("gpt2")
gpt2_ids = enc.encode(corpus)
vocab = sorted(set(gpt2_ids))               # re-number only the tokens we use (as in lesson 13)
to_local = {g: i for i, g in enumerate(vocab)}
data = torch.tensor([to_local[g] for g in gpt2_ids])

config = GPTConfig(vocab_size=len(vocab), block_size=16, n_layer=2, n_head=2, n_embd=32)
model = GPT2(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
batch_size = 16


def get_batch():
    """Grab batch_size random chunks of the stream, and the same chunks shifted by one."""
    starts = torch.randint(len(data) - config.block_size - 1, (batch_size,))
    x = torch.stack([data[s:s + config.block_size] for s in starts])
    y = torch.stack([data[s + 1:s + config.block_size + 1] for s in starts])
    return x, y


print(f"TRAINING a {sum(p.numel() for p in model.parameters()):,}-parameter GPT-2 on "
      f"{len(data)} tokens (vocabulary of {len(vocab)})")
for step in range(301):
    xb, yb = get_batch()
    _, loss = model(xb, yb)                 # stages 1-4
    optimizer.zero_grad()
    loss.backward()                         # stage 5
    optimizer.step()                        # stage 6
    if step % 50 == 0:
        print(f"  step {step:>3}: loss = {loss.item():.3f}")

checkpoint = Path(__file__).parent / "checkpoints" / "tiny_gpt.pt"
checkpoint.parent.mkdir(exist_ok=True)
torch.save({"config": vars(config), "vocab": vocab, "model": model.state_dict()}, checkpoint)
print(f"\nSaved the trained model to {checkpoint} - lesson 30 generates text with it.")
