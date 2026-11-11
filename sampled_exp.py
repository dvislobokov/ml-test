"""Эксперимент 5: грубый (прореженный) обратный проход под Muon и под AdamW.

Градиент весов линейного слоя — это произведение ΔᵀA по всем токенам батча.
Здесь он оценивается по случайной доле последовательностей батча (своей для
каждого слоя), а градиент по входу считается честно, чтобы сигнал дошёл до
нижних слоёв. Гипотеза: Muon оставляет от градиента только «направления» и
потому должен терпеть такую грубость лучше AdamW.

Обязательный контроль — просто меньший батч: если прореживание не лучше него
при равном времени, идея бесполезна.

Все варианты получают одинаковый бюджет времени обучения; шаг затухает
линейно по доле израсходованного времени.

  python3 sampled_exp.py muon 32 1.0  45 0    # базовый
  python3 sampled_exp.py muon 32 0.25 45 0    # градиент весов по 1/4 батча
  python3 sampled_exp.py muon 16 1.0  45 0    # контроль: батч вдвое меньше
"""
import json
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import MiniGPT, load_text, text_batch
from muon_exp import evaluate


class SampledLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, k):
        ctx.save_for_backward(x, W)
        ctx.k = k
        return F.linear(x, W)

    @staticmethod
    def backward(ctx, g):
        x, W = ctx.saved_tensors
        B, k = x.shape[0], ctx.k
        gx = g @ W
        s = int(torch.randint(0, B - k + 1, (1,))) if k < B else 0
        gs = g[s : s + k].reshape(-1, g.shape[-1])
        xs = x[s : s + k].reshape(-1, x.shape[-1])
        gW = gs.T @ xs
        if k < B:
            gW = gW * (B / k)  # несмещённая оценка
        return gx, gW, None


def patch(model, batch, frac):
    k = max(1, round(batch * frac))
    for blk in model.blocks:
        for name in ("qkv", "proj", "fc", "out"):
            lin = getattr(blk, name)
            lin.forward = (lambda x, lin=lin: SampledLinearFn.apply(x, lin.weight, k))
    return k


def main(optname, batch, frac, budget, seed, every_s=5.0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    tr, va, vocab = load_text()
    model = MiniGPT(vocab)
    k = patch(model, batch, frac)
    vg = torch.Generator().manual_seed(123)
    val = [text_batch(va, 32, 128, vg) for _ in range(20)]
    loss_fn = lambda out, y: F.cross_entropy(out.view(-1, out.shape[-1]), y.view(-1))

    named = list(model.named_parameters())
    is_mu = lambda n, p: p.ndim == 2 and n.startswith("blocks.")
    if optname == "muon":
        opts = [
            torch.optim.Muon([p for n, p in named if is_mu(n, p)], lr=0.03, weight_decay=0.0),
            torch.optim.AdamW([p for n, p in named if not is_mu(n, p)], lr=3e-3, weight_decay=0.0),
        ]
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)]
    base = [[gp["lr"] for gp in o.param_groups] for o in opts]

    log, t_train, steps, next_eval = [], 0.0, 0, 0.0
    while True:
        done = t_train >= budget
        if t_train >= next_eval or done:
            vl, acc = evaluate(model, val, loss_fn)
            log.append({"step": steps, "time": t_train, "val_loss": vl, "val_acc": acc})
            next_eval += every_s
        if done:
            break
        t0 = time.perf_counter()
        fr = max(0.0, 1.0 - t_train / budget)
        for o, b in zip(opts, base):
            for gp, lr0 in zip(o.param_groups, b):
                gp["lr"] = lr0 * fr
        x, y = text_batch(tr, batch, 128, g)
        loss = loss_fn(model(x), y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in opts:
            o.step()
        t_train += time.perf_counter() - t0
        steps += 1
    r = {"opt": optname, "batch": batch, "frac": frac, "k": k, "budget": budget, "seed": seed,
         "steps": steps, "ms_per_step": 1e3 * t_train / steps, "log": log}
    print(f"RESULT {optname} batch={batch} frac={frac} seed={seed} steps={steps} "
          f"ms/step={r['ms_per_step']:.1f} val_loss={log[-1]['val_loss']:.4f}", flush=True)
    return r


if __name__ == "__main__":
    optname, batch, frac, budget, seed = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    r = main(optname, batch, frac, budget, seed)
    if budget >= 30:
        with open("results/sampled_runs.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
