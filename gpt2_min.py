"""
The complete GPT-2 from the "Transformer is easy" video, in about 100 lines.

Lessons 20-30 take it apart one piece at a time:
  GPTConfig            the blueprint: all the size knobs                  lesson 20
  CausalSelfAttention  communication - tokens share information           lessons 21-24
  MLP                  thinking - each token processes on its own         lesson 25
  Block                LayerNorm + residuals glue the two together        lessons 26-27
  GPT2                 embeddings -> stack of blocks -> LM head           lessons 28-30
Shared by those lessons so the model code lives in one place (like tiny_net.py).
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int = 50257             # how many distinct tokens the model knows
    block_size: int = 1024              # context window: how many tokens it can see at once
    n_layer: int = 12                   # depth: how many transformer blocks are stacked
    n_head: int = 12                    # attention heads per block (parallel "specialists")
    n_embd: int = 768                   # embedding dimension: numbers per token vector
    dropout: float = 0.0                # regularization; 0.0 switches it off


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention: each token gathers information from earlier tokens."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0   # every head gets an equal slice of C
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # fused Q, K, V projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # mixes the heads' results
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        # Lower-triangular causal mask. A buffer: saved and moved with the model, never trained.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()                          # batch, time (sequence length), channels
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)          # each (B, T, C)
        hs = C // self.n_head                       # head size
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)           # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)             # scores (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))   # hide the future
        att = F.softmax(att, dim=-1)                # each row becomes weights that sum to 1
        att = self.attn_dropout(att)
        y = att @ v                                 # weighted sum of values (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # merge the heads
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Expand to 4*C, GELU, contract back to C - applied to every token independently."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)     # expand
        self.gelu = nn.GELU()                                       # non-linearity
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)   # contract
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One transformer block: normalize, process, add - once for attention, once for the MLP."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))             # communicate (pre-norm + residual)
        x = x + self.mlp(self.ln_2(x))              # think      (pre-norm + residual)
        return x


class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)   # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)   # position embeddings
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)                     # final stabilizer
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight       # weight tying: one matrix, two jobs
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # GPT-2 starts every weight as small noise (std 0.02). PyTorch's default of std 1 for
        # embeddings would make the first predictions wildly over-confident.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        assert T <= self.config.block_size, "sequence is longer than the context window"
        pos = torch.arange(0, T, device=idx.device)                 # 0, 1, ..., T-1
        x = self.drop(self.wte(idx) + self.wpe(pos))                # (B, T, C)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                    # (B, T, vocab_size): a guess at every position
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]             # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature                 # keep only the last guess
            if top_k is not None:
                kth_best = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits[logits < kth_best] = float("-inf")           # drop all but the top k
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)      # roll the dice
            idx = torch.cat((idx, idx_next), dim=1)                 # append and repeat
        return idx
