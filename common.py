"""Общие модели и данные для экспериментов по скорости обучения."""
import gzip
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------- данные
def _idx(name):
    with gzip.open(DATA / name, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8)


def load_mnist():
    """Возвращает (xtr, ytr, xte, yte): float32 [N,1,28,28], нормированные."""
    xtr = _idx("train-images-idx3-ubyte.gz")[16:].reshape(-1, 1, 28, 28)
    ytr = _idx("train-labels-idx1-ubyte.gz")[8:]
    xte = _idx("t10k-images-idx3-ubyte.gz")[16:].reshape(-1, 1, 28, 28)
    yte = _idx("t10k-labels-idx1-ubyte.gz")[8:]
    f = lambda x: (torch.from_numpy(x.copy()).float() / 255.0 - 0.1307) / 0.3081
    g = lambda y: torch.from_numpy(y.copy()).long()
    return f(xtr), g(ytr), f(xte), g(yte)


def load_text():
    """Посимвольный корпус: (train_ids, val_ids, vocab_size)."""
    text = (DATA / "tinyshakespeare.txt").read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


def text_batch(ids, batch, ctx, gen=None):
    ix = torch.randint(len(ids) - ctx - 1, (batch,), generator=gen)
    x = torch.stack([ids[i : i + ctx] for i in ix])
    y = torch.stack([ids[i + 1 : i + ctx + 1] for i in ix])
    return x, y


# ---------------------------------------------------------------- модели
class MLP(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class MiniResNet(nn.Module):
    def __init__(self, w=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU()
        )
        self.s1 = Block(w, w, 1)
        self.s2 = Block(w, 2 * w, 2)
        self.s3 = Block(2 * w, 4 * w, 2)
        self.fc = nn.Linear(4 * w, 10)

    def forward(self, x):
        x = self.s3(self.s2(self.s1(self.stem(x))))
        return self.fc(x.mean((2, 3)))


class GPTBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d, bias=False)
        self.out = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, -1).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class MiniGPT(nn.Module):
    def __init__(self, vocab, d=128, layers=4, heads=4, ctx=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(GPTBlock(d, heads) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def cosine_lr(step, total, base, warmup=0):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
