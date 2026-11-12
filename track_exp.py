"""Эксперимент 7: следящая ортогонализация (TrackMuon) против Muon, равный бюджет времени.

  python3 track_exp.py muon  8 0.03 45 0            # встроенный torch.optim.Muon
  python3 track_exp.py muon3 8 0.03 45 0            # он же с 3 итерациями вместо 5
  python3 track_exp.py track 8 0.03 45 0 0.3        # TrackMuon, alpha=0.3
  python3 track_exp.py track 8 0.03 45 0 0.3 0.5 2  # ... beta=0.5, 2 итерации
Аргументы: вариант, батч, шаг обучения, бюджет времени (с), seed [, alpha, beta, iters, refresh].
"""
import json
import sys
import time

import torch
import torch.nn.functional as F

from common import MiniGPT, load_text, text_batch
from muon_exp import evaluate
from track_muon import TrackMuon


def main(variant, batch, lr, budget, seed, alpha=0.3, beta=0.0, iters=1, refresh=0, every_s=5.0):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    tr, va, vocab = load_text()
    model = MiniGPT(vocab)
    vg = torch.Generator().manual_seed(123)
    val = [text_batch(va, 32, 128, vg) for _ in range(20)]
    loss_fn = lambda out, y: F.cross_entropy(out.view(-1, out.shape[-1]), y.view(-1))

    named = list(model.named_parameters())
    is_mu = lambda n, p: p.ndim == 2 and n.startswith("blocks.")
    mu = [p for n, p in named if is_mu(n, p)]
    rest = [p for n, p in named if not is_mu(n, p)]
    if variant.startswith("muon"):
        mopt = torch.optim.Muon(mu, lr=lr, weight_decay=0.0, ns_steps=int(variant[4:] or 5))
    else:
        mopt = TrackMuon(mu, lr=lr, alpha=alpha, beta=beta, iters=iters, refresh=refresh)
    opts = [mopt, torch.optim.AdamW(rest, lr=3e-3, weight_decay=0.0)]
    base = [[gp["lr"] for gp in o.param_groups] for o in opts]

    log, t_train, t_opt, steps, next_eval = [], 0.0, 0.0, 0, 0.0
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
        t1 = time.perf_counter()
        for o in opts:
            o.step()
        t2 = time.perf_counter()
        t_train += t2 - t0
        t_opt += t2 - t1
        steps += 1
    r = {"variant": variant, "batch": batch, "lr": lr, "budget": budget, "seed": seed,
         "alpha": alpha, "beta": beta, "iters": iters, "refresh": refresh, "steps": steps,
         "ms_per_step": 1e3 * t_train / steps, "opt_ms_per_step": 1e3 * t_opt / steps, "log": log}
    extra = "" if variant.startswith("muon") else f" alpha={alpha} beta={beta} iters={iters} refresh={refresh}"
    print(f"RESULT {variant} batch={batch} lr={lr}{extra} seed={seed} steps={steps} "
          f"ms/step={r['ms_per_step']:.1f} opt={r['opt_ms_per_step']:.1f} val_loss={log[-1]['val_loss']:.4f}", flush=True)
    return r


if __name__ == "__main__":
    a = sys.argv
    r = main(a[1], int(a[2]), float(a[3]), float(a[4]), int(a[5]),
             float(a[6]) if len(a) > 6 else 0.3, float(a[7]) if len(a) > 7 else 0.0, int(a[8]) if len(a) > 8 else 1, int(a[9]) if len(a) > 9 else 0)
    if r["budget"] >= 30:
        with open("results/track_runs.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
