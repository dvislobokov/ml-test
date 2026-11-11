"""Эксперимент 2: Muon против AdamW — по шагам и по реальному времени.

Muon применяется только к двумерным весам скрытых слоёв; эмбеддинги, выходной слой,
нормализации и смещения остаются на AdamW (как в оригинальном рецепте).
Время считается только по шагам обучения (оценка на валидации исключена).

  python3 muon_exp.py gpt adamw 3e-3 600 0
  python3 muon_exp.py gpt muon  0.02 600 0
"""
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from common import MLP, MiniGPT, load_mnist, load_text, text_batch


AMP = os.environ.get("AMP") == "1"  # прямой и обратный проход в bfloat16


def build(task, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    if task == "gpt":
        tr, va, vocab = load_text()
        model = MiniGPT(vocab)
        vg = torch.Generator().manual_seed(123)
        val = [text_batch(va, 32, 128, vg) for _ in range(20)]
        batch = lambda: text_batch(tr, 32, 128, g)
        loss_fn = lambda out, y: F.cross_entropy(out.view(-1, out.shape[-1]), y.view(-1))
        muon_names = lambda n, p: p.ndim == 2 and n.startswith("blocks.")
    else:
        xtr, ytr, xte, yte = load_mnist()
        model = MLP()
        val = [(xte, yte)]

        def batch():
            ix = torch.randint(len(xtr), (128,), generator=g)
            return xtr[ix], ytr[ix]

        loss_fn = lambda out, y: F.cross_entropy(out, y)
        muon_names = lambda n, p: n in ("fc1.weight", "fc2.weight")
    return model, batch, val, loss_fn, muon_names


@torch.no_grad()
def evaluate(model, val, loss_fn):
    model.eval()
    loss = acc = 0.0
    for x, y in val:
        out = model(x)
        loss += loss_fn(out, y).item()
        acc += (out.argmax(-1) == y).float().mean().item()
    model.train()
    return loss / len(val), acc / len(val)


def main(task, optname, lr, steps, seed, every=50):
    model, batch, val, loss_fn, is_muon = build(task, seed)
    named = list(model.named_parameters())
    if optname == "muon":
        mu = [p for n, p in named if is_muon(n, p)]
        rest = [p for n, p in named if not is_muon(n, p)]
        opts = [
            torch.optim.Muon(mu, lr=lr, weight_decay=0.0, momentum=0.95),
            torch.optim.AdamW(rest, lr=3e-3 if task == "gpt" else 1e-3, weight_decay=0.0),
        ]
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)]
    base = [[g["lr"] for g in o.param_groups] for o in opts]

    log, t_train, t_opt = [], 0.0, 0.0
    for s in range(steps + 1):
        if s % every == 0:
            vl, va = evaluate(model, val, loss_fn)
            log.append({"step": s, "time": t_train, "val_loss": vl, "val_acc": va})
            print(f"{task} {optname} lr={lr} step {s:5d} t={t_train:7.1f}s val_loss={vl:.4f} acc={va:.4f}", flush=True)
        if s == steps:
            break
        t0 = time.perf_counter()
        frac = 1.0 - s / steps  # линейное затухание шага до нуля
        for o, b in zip(opts, base):
            for gp, lr0 in zip(o.param_groups, b):
                gp["lr"] = lr0 * frac
        x, y = batch()
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=AMP):
            loss = loss_fn(model(x), y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        t1 = time.perf_counter()
        for o in opts:
            o.step()
        t2 = time.perf_counter()
        t_train += t2 - t0
        t_opt += t2 - t1
    return {
        "task": task, "opt": optname + ("+bf16" if AMP else ""), "lr": lr, "steps": steps, "seed": seed,
        "ms_per_step": 1e3 * t_train / steps, "opt_ms_per_step": 1e3 * t_opt / steps,
        "log": log,
    }


if __name__ == "__main__":
    task, optname, lr, steps, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    r = main(task, optname, lr, steps, seed)
    with open("results/muon_runs.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
