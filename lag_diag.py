"""Диагностика: согласованы ли градиенты соседних шагов (косинус между G_t и G_{t+k}).

Без выбора направлений, поэтому без смещения отбора. Независимый шум + слабый сигнал
дают малый положительный косинус на всех лагах; «дребезг» с периодом 2 — отрицательный
на нечётных лагах и положительный на чётных.

  python3 lag_diag.py muon 8 0.03
"""
import sys
from collections import deque

import torch
import torch.nn.functional as F

from common import MiniGPT, load_text, text_batch


def main(optname, batch, lr, steps=300, start=100, lags=(1, 2, 3, 4)):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    tr, _, vocab = load_text()
    model = MiniGPT(vocab)
    loss_fn = lambda out, y: F.cross_entropy(out.view(-1, out.shape[-1]), y.view(-1))
    named = list(model.named_parameters())
    is_mu = lambda n, p: p.ndim == 2 and n.startswith("blocks.")
    mu = [p for n, p in named if is_mu(n, p)]
    if optname == "muon":
        opts = [torch.optim.Muon(mu, lr=lr, weight_decay=0.0),
                torch.optim.AdamW([p for n, p in named if not is_mu(n, p)], lr=3e-3, weight_decay=0.0)]
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)]
    hist = deque(maxlen=max(lags) + 1)
    acc = {k: [] for k in lags}
    for s in range(steps):
        x, y = text_batch(tr, batch, 128, g)
        loss = loss_fn(model(x), y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        cur = [p.grad.clone() for p in mu]
        if s >= start:
            for k in lags:
                if len(hist) >= k:
                    old = hist[-k]
                    cs = [(a * b).sum() / (a.norm() * b.norm()) for a, b in zip(cur, old)]
                    acc[k].append(torch.stack(cs).mean().item())
        hist.append(cur)
        for o in opts:
            o.step()
    print(f"{optname} батч={batch} lr={lr}: средний косинус G_t и G_(t+k) по матрицам блоков: "
          + "  ".join(f"k={k}: {sum(v)/len(v):+.3f}" for k, v in acc.items()))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
