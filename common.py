"""Shared pieces for parts 3-5: the model, the data, the schedules, the plot style.

One 4-layer decoder over TinyShakespeare characters. `d_model` is the width knob
(256/512/1024); head_dim is pinned at 64 so widening adds heads, not head size.
Standard parameterization on purpose -- part 5 is about how the optimal LR moves
with width, which is exactly what muP would remove.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ, BATCH, N_LAYER, HEAD_DIM = 128, 32, 4, 64
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "tinyshakespeare.txt")

# dataviz reference palette, categorical slots 1-3 (validated all-pairs, light).
C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# blue sequential ramp, ordinal floor step 250 -> step 700, for depth-ordered layers.
BLUE_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
SURFACE, INK, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e"


def device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def style():
    """Chart style: recessive axes, light surface, ink-colored text."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
        "axes.axisbelow": True, "figure.dpi": 150, "font.size": 9,
        "legend.frameon": False, "lines.linewidth": 2.0,
    })
    return plt


# ---------------------------------------------------------------- data

class Data:
    """Character-level TinyShakespeare, 90/10 split, deterministic batches."""

    def __init__(self, dev):
        text = open(DATA).read()
        self.vocab = sorted(set(text))
        stoi = {c: i for i, c in enumerate(self.vocab)}
        ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(ids))
        self.train, self.val = ids[:n].to(dev), ids[n:].to(dev)
        self.dev = dev

    def batch(self, split, gen):
        d = self.train if split == "train" else self.val
        ix = torch.randint(len(d) - SEQ - 1, (BATCH,), generator=gen, device=self.dev)
        x = torch.stack([d[i:i + SEQ] for i in ix])
        y = torch.stack([d[i + 1:i + SEQ + 1] for i in ix])
        return x, y


# ---------------------------------------------------------------- model

class Transformer(nn.Module):
    def __init__(self, vocab, d_model):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(SEQ, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead=d_model // HEAD_DIM, dim_feedforward=4 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, N_LAYER, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, x, y=None):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.blocks(h, mask=mask, is_causal=True)
        logits = self.head(self.norm(h))
        if y is None:
            return logits
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


def build(data, d_model, seed=0):
    torch.manual_seed(seed)
    return Transformer(len(data.vocab), d_model).to(data.dev)


# ---------------------------------------------------------------- schedules

def lr_at(step, total, lr, kind="cosine", warmup=20, decay_frac=0.2, floor=0.0):
    """Multiplier schedules. `step` is 0-indexed.

    cosine: warmup then cosine to `floor` over the whole run.
    wsd:    warmup, then constant, then linear decay over the last `decay_frac`.
    const:  warmup then flat (used for the anneal-from-a-checkpoint branch).
    """
    if step < warmup:
        return lr * (step + 1) / warmup
    if kind == "const":
        return lr
    p = (step - warmup) / max(1, total - warmup)
    if kind == "cosine":
        return floor + (lr - floor) * 0.5 * (1 + math.cos(math.pi * p))
    if kind == "wsd":
        stable = 1 - decay_frac
        return lr if p < stable else floor + (lr - floor) * (1 - p) / decay_frac
    raise ValueError(kind)


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model, data, n_batch=20, seed=1234):
    """Val loss on a fixed batch set -- same batches for every run, so runs compare."""
    model.eval()
    gen = torch.Generator(device=data.dev).manual_seed(seed)
    loss = sum(model(*data.batch("val", gen)).item() for _ in range(n_batch)) / n_batch
    model.train()
    return loss


def train(model, data, steps, lr, kind="cosine", warmup=20, decay_frac=0.2,
          seed=0, betas=(0.9, 0.95), wd=0.1, on_step=None, eval_every=0,
          opt=None, gen=None, start=0, total=None):
    """Adam(W) training loop. Returns (train_losses, eval_points, opt, gen).

    Data order is fixed by `seed` so two schedules see identical batches.
    `on_step(t, model, prev, lr)` runs after each step (part 3 uses it for ratios).
    Pass back `opt`/`gen` with `start`/`total` to resume a run where it stopped --
    part 4 needs that to anneal a mid-run checkpoint without repeating the run.
    """
    opt = opt or torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
    gen = gen or torch.Generator(device=data.dev).manual_seed(seed)
    total = total or steps
    losses, evals = [], []
    for t in range(start, start + steps):
        cur = lr_at(t, total, lr, kind, warmup, decay_frac)
        for g in opt.param_groups:
            g["lr"] = cur
        loss = model(*data.batch("train", gen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        prev = {n: p.detach().clone() for n, p in model.named_parameters()} if on_step else None
        opt.step()
        losses.append(loss.item())
        if not math.isfinite(losses[-1]):
            break          # a diverged LR is decided; don't pay for the rest of the run
        if on_step:
            on_step(t, model, prev, cur)
        if eval_every and (t + 1) % eval_every == 0:
            evals.append((t + 1, evaluate(model, data)))
    return losses, evals, opt, gen
