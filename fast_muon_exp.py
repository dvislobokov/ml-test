"""Эксперимент 3: удешевлённая ортогонализация в Muon (см. fast_muon.py).

  python3 fast_muon_exp.py mlp cold-5 0.01 1000 0
  python3 fast_muon_exp.py gpt warm   0.03 500 0
Варианты: cold-N, warm, lazy-K.
"""
import json
import sys
import time

import torch

from fast_muon import FastMuon, polar_quality
from muon_exp import build, evaluate


def make_opt(params, variant, lr):
    kind, _, arg = variant.partition("-")
    if kind == "cold":
        return FastMuon(params, lr=lr, mode="cold", ns_steps=int(arg or 5))
    if kind == "lazy":
        return FastMuon(params, lr=lr, mode="lazy", lazy_k=int(arg or 4))
    return FastMuon(params, lr=lr, mode="warm")


def main(task, variant, lr, steps, seed, every=50, diag_every=100):
    model, batch, val, loss_fn, is_muon = build(task, seed)
    named = list(model.named_parameters())
    mu = [p for n, p in named if is_muon(n, p)]
    rest = [p for n, p in named if not is_muon(n, p)]
    muon = make_opt(mu, variant, lr)
    opts = [muon, torch.optim.AdamW(rest, lr=3e-3 if task == "gpt" else 1e-3, weight_decay=0.0)]
    base = [[g["lr"] for g in o.param_groups] for o in opts]
    probe = max(mu, key=lambda p: p.numel())  # самая большая матрица — для диагностики

    log, diag, t_train, t_opt = [], [], 0.0, 0.0
    for s in range(steps + 1):
        if s % every == 0:
            vl, va = evaluate(model, val, loss_fn)
            log.append({"step": s, "time": t_train, "val_loss": vl, "val_acc": va})
            print(f"{task} {variant} step {s:5d} t={t_train:7.1f}s val_loss={vl:.4f} acc={va:.4f}", flush=True)
        if s == steps:
            break
        t0 = time.perf_counter()
        frac = 1.0 - s / steps
        for o, b in zip(opts, base):
            for gp, lr0 in zip(o.param_groups, b):
                gp["lr"] = lr0 * frac
        x, y = batch()
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
        if s % diag_every == diag_every - 1:  # вне измеряемого времени
            cos, smin, smax = polar_quality(*muon.last[id(probe)])
            diag.append({"step": s, "cos": cos, "sv_min": smin, "sv_max": smax})
    r = {
        "task": task, "variant": variant, "lr": lr, "steps": steps, "seed": seed,
        "ms_per_step": 1e3 * t_train / steps, "opt_ms_per_step": 1e3 * t_opt / steps,
        "orth_calls": muon.stats, "diag": diag, "log": log,
    }
    d = diag[-1]
    print(f"RESULT {task} {variant} seed={seed} ms/step={r['ms_per_step']:.1f} opt={r['opt_ms_per_step']:.1f} "
          f"val_loss={log[-1]['val_loss']:.4f} acc={log[-1]['val_acc']:.4f} "
          f"cos={sum(x['cos'] for x in diag)/len(diag):.3f} sv=[{d['sv_min']:.2f},{d['sv_max']:.2f}] {muon.stats}", flush=True)
    return r


if __name__ == "__main__":
    task, variant, lr, steps, seed = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    r = main(task, variant, lr, steps, seed)
    if steps >= 300:
        with open("results/fast_muon_runs.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
