"""
Lesson 20 - A tiny transformer: lesson 13 with the black box opened.

Lesson 13's whole model was one line:   logits = W[X]   (look at the LAST token only)
Here that line becomes a real transformer, which looks at ALL previous tokens:

    token ids
      -> token embedding + position embedding      (what each token is, and where it is)
      -> self-attention                            (each token gathers info from earlier tokens)
      -> feed-forward network                      (each token "thinks" about what it gathered)
      -> linear head                               (one logit per vocabulary token)

Everything around the model is unchanged: same corpus, same GPT-2 tokens, same
softmax + cross-entropy loss, same backward + update loop, same temperature / top-p sampling.
The only practical difference: autograd (lesson 15) computes the gradients, because writing
backprop by hand for attention would be pages of algebra.
"""

import math
from collections import Counter

import tiktoken
import torch
import torch.nn as nn

torch.manual_seed(0)                            # reproducible weights, so every run prints the same

# ---------------------------------------------------------------------------
# 1-3. Data, tokenization, vocabulary: identical to lesson 13.
# ---------------------------------------------------------------------------
corpus = """The cat sat on the mat. The dog sat on the rug. The cat chased the dog.
The dog chased the ball. The cat slept on the mat. The dog slept on the rug.
In Paris the people speak French. In Berlin the people speak German.
The weather today is sunny. The weather today is cloudy. The weather today is cold."""

enc = tiktoken.get_encoding("gpt2")
gpt2_ids = enc.encode(corpus)                   # text -> GPT-2 token IDs
vocab = sorted(set(gpt2_ids))                   # only the tokens our corpus uses
to_local = {g: i for i, g in enumerate(vocab)}  # GPT-2 ID -> compact ID 0..V-1
V = len(vocab)                                  # vocabulary size (28)
ids = torch.tensor([to_local[g] for g in gpt2_ids])   # the corpus as compact IDs
print(f"Corpus: {len(ids)} tokens, vocabulary of {V}")

# ---------------------------------------------------------------------------
# Training examples. Lesson 13 used (one token -> next token) pairs.
# A transformer gets a whole window of context: for a window of `block_size` tokens it makes
# block_size predictions at once - token 0 predicts token 1, tokens 0-1 predict token 2, ...
# (the "input so far -> next token" pairs from lesson 08, all in one shot).
# ---------------------------------------------------------------------------
block_size = 16                                 # max context length the model can look back over
starts = range(len(ids) - block_size)           # every possible window start position
X = torch.stack([ids[s:s + block_size] for s in starts])          # inputs,  shape (59, 16)
Y = torch.stack([ids[s + 1:s + 1 + block_size] for s in starts])  # targets: the same windows shifted by one
print(f"Training windows: {tuple(X.shape)}  (windows x tokens)\n")


# ---------------------------------------------------------------------------
# 4. The model.
# ---------------------------------------------------------------------------
class SelfAttention(nn.Module):
    """One attention head, written out step by step.

    Every token produces three vectors:
      query - "what am I looking for?"
      key   - "what do I contain?"
      value - "what will I hand over if someone attends to me?"
    A token's query is compared to every earlier token's key; the better the match, the more
    of that token's value it takes in. This is how "speak" can reach back to "Berlin".
    """

    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)  # each is a learnable matrix (lesson 18)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)                # mixes the result back into the stream
        self.last_weights = None                              # kept so we can peek at them later

    def forward(self, x):                       # x: (batch, T, d_model)
        T, d = x.shape[1], x.shape[2]
        q, k, v = self.query(x), self.key(x), self.value(x)   # three projections of every token
        scores = q @ k.transpose(-2, -1) / math.sqrt(d)       # (batch, T, T): every query vs every key;
                                                              # dividing by sqrt(d) keeps scores moderate
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))   # lower triangle = "earlier or same"
        scores = scores.masked_fill(~causal, float("-inf"))   # no peeking at FUTURE tokens (that's the answer!)
        weights = torch.softmax(scores, dim=-1)               # each row -> attention percentages summing to 1
        self.last_weights = weights.detach()
        return self.out(weights @ v)                          # weighted average of the values


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, block_size, d_model=32, d_hidden=128):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)    # token ID   -> vector ("what")
        self.pos_emb = nn.Embedding(block_size, d_model)      # position i -> vector ("where")
        self.ln1 = nn.LayerNorm(d_model)                      # keep numbers in a stable range (lesson 18)
        self.attn = SelfAttention(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(                             # the feed-forward network from lesson 19
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)            # vector -> one logit per vocabulary token

    def forward(self, idx):                     # idx: (batch, T) token IDs
        T = idx.shape[1]
        positions = torch.arange(T)             # 0, 1, ..., T-1
        x = self.token_emb(idx) + self.pos_emb(positions)     # without positions, word order would be invisible
        x = x + self.attn(self.ln1(x))          # "x +": a residual connection - attention ADDS information
        x = x + self.ffn(self.ln2(x))           #   to each token's vector instead of replacing it
        return self.head(self.ln_final(x))      # logits: (batch, T, vocab_size)


model = TinyTransformer(V, block_size)
n_params = sum(p.numel() for p in model.parameters())
print(f"Tiny transformer: {n_params:,} parameters  (lesson 13's bigram table: {V * V:,})")

# ---------------------------------------------------------------------------
# 5. The training loop - the same five steps as lesson 13.
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)   # Adam (lesson 19), with weight decay
for step in range(301):
    logits = model(X)                           # FORWARD: this is the line that replaced  W[X]
    log_probs = torch.log_softmax(logits, dim=-1)              # log(softmax), computed stably in one go
    loss = -log_probs.gather(-1, Y.unsqueeze(-1)).mean()       # LOSS: -log p(correct), via gather (lesson 16)
    optimizer.zero_grad()                       # reset
    loss.backward()                             # BACKPROP: autograd does the chain rule for every weight
    optimizer.step()                            # GRADIENT DESCENT: nudge every weight
    if step in (0, 10, 50, 100, 300):
        print(f"  step {step:>3}: loss = {loss.item():.3f}")

model.eval()                                    # switch to evaluation mode (good habit; no dropout here)


# ---------------------------------------------------------------------------
# 6. Fair comparison with the bigram model, on exactly lesson 13's 74 next-token predictions.
# ---------------------------------------------------------------------------
@torch.no_grad()                                # no gradients needed when just measuring
def next_token_probs(context_ids):
    """Probabilities for the token after `context_ids` (uses up to block_size tokens of context)."""
    ctx = torch.tensor(context_ids[-block_size:]).unsqueeze(0)   # keep the last 16, add a batch dim
    return torch.softmax(model(ctx)[0, -1], dim=-1)             # the prediction at the LAST position


ids_list = ids.tolist()
transformer_loss = sum(-math.log(next_token_probs(ids_list[:t])[ids_list[t]])
                       for t in range(1, len(ids_list))) / (len(ids_list) - 1)

# The best ANY bigram model can do: use the true counts, p(b | a) = count(a, b) / count(a).
pair_counts = Counter(zip(ids_list, ids_list[1:]))
first_counts = Counter(ids_list[:-1])
best_bigram_loss = sum(-math.log(pair_counts[(a, b)] / first_counts[a])
                       for a, b in zip(ids_list, ids_list[1:])) / (len(ids_list) - 1)

print("\nAverage cross-entropy over the corpus's 74 next-token predictions:")
print(f"  bigram, lesson 13 after 300 steps : 0.667")
print(f"  bigram, best possible (counting)  : {best_bigram_loss:.3f}")
print(f"  tiny transformer                  : {transformer_loss:.3f}")


# ---------------------------------------------------------------------------
# 7. The test: Berlin -> German.
# ---------------------------------------------------------------------------
def show_next(prompt):
    context = [to_local[g] for g in enc.encode(prompt)]
    p = next_token_probs(context)
    top = torch.argsort(p, descending=True)[:3]
    print(f"  {prompt!r:<32} -> " + ", ".join(f"{enc.decode([vocab[i]])!r} {p[i]:.0%}" for i in top))


speak_id = to_local[enc.encode(" speak")[0]]
german_id, french_id = to_local[enc.encode(" German")[0]], to_local[enc.encode(" French")[0]]
n_speak = first_counts[speak_id]
print("\nAfter '... speak', the bigram model can only see 'speak':")
print(f"  bigram (best possible): ' German' {pair_counts[(speak_id, german_id)] / n_speak:.0%}, "
      f"' French' {pair_counts[(speak_id, french_id)] / n_speak:.0%}  - a coin flip")
# Careful with spaces (lesson 09): GPT-2 tokens carry their leading space. In the corpus,
# "In Paris" starts a line -> token 'In', but "In Berlin" follows "French." -> token ' In'.
# These are two DIFFERENT tokens, so we prompt each sentence with the token it really starts with.
print("The transformer sees the whole sentence:")
show_next(" In Berlin the people speak")
show_next("In Paris the people speak")

# What if we write 'In' without the space before Berlin? In training, 'In' was ALWAYS followed
# by ' Paris', so this prompt mixes signals from both sentences and the answer becomes shaky
# (the exact split depends on the random seed).
print("Same words, but Berlin prompted with Paris's 'In' token (never seen in training):")
show_next("In Berlin the people speak")

# Peek inside: where did 'speak' look? Row = the last token's attention over the context.
prompt = " In Berlin the people speak"
ctx = [to_local[g] for g in enc.encode(prompt)]
with torch.no_grad():
    model(torch.tensor(ctx).unsqueeze(0))       # forward pass just to fill attn.last_weights
weights = model.attn.last_weights[0, -1]        # attention of the final token ('speak') over all 5 tokens
print(f"\nAttention from ' speak' in {prompt!r}:")
for tok, w in zip(ctx, weights):
    print(f"  {enc.decode([vocab[tok]])!r:<10} {w:6.1%} " + "#" * round(w.item() * 40))


# ---------------------------------------------------------------------------
# 8. Generation - lesson 13's sampler, now feeding the transformer the whole context.
# ---------------------------------------------------------------------------
def generate(prompt, n_tokens, temperature=1.0, top_p=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = [to_local[t] for t in enc.encode(prompt)]
    for _ in range(n_tokens):
        ctx = torch.tensor(out[-block_size:]).unsqueeze(0)    # crop to the last block_size tokens
        with torch.no_grad():
            logits = model(ctx)[0, -1] / temperature           # temperature (lesson 12)
        p = torch.softmax(logits, dim=-1)
        p_sorted, order = torch.sort(p, descending=True)       # top-p (lesson 12)
        cutoff = int(torch.searchsorted(torch.cumsum(p_sorted, 0), torch.tensor(top_p))) + 1
        nucleus = p_sorted[:cutoff] / p_sorted[:cutoff].sum()
        out.append(order[torch.multinomial(nucleus, 1, generator=g)].item())
    return enc.decode([vocab[i] for i in out])


print("\nGeneration (T=1.0, top_p=0.9):")
for prompt in [" In Berlin", "In Paris", "The weather"]:
    print(f"  {generate(prompt, 10, 1.0, 0.9)!r}")
print(f"\nCaveat: with 75 tokens of training text and {n_params:,} parameters, the model can largely")
print("MEMORISE the corpus - that's why its loss is so low. Real LLMs train on trillions of")
print("tokens so memorising is impossible and they must learn general patterns instead.")
