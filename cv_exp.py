"""Эксперимент 6: CV-Muon против обычного Muon (мини-GPT), сравнение по шагам.

  python3 cv_exp.py muon  8 0.03 1200 0     # встроенный torch.optim.Muon
  python3 cv_exp.py exact 8 0.03 1200 0     # контроль: точный полярный множитель
  python3 cv_exp.py v2    8 0.05 1200 0     # CV-Muon, честная перекрёстная проверка
Аргументы: вариант, батч, шаг обучения, число шагов, seed.
"""
import json
import sys
import time

import torch
import torch.nn.functional as F

from common import MiniGPT, load_text, text_batch
from cv_muon import CVMuon
from muon_exp import evaluate


def gain_profile(opt, params):
    """Средний коэффициент по четвертям спектра (от крупных сингулярных чисел к мелким)."""
    q = torch.zeros(4)
    k = 0
    for p in params:
        if id(p) not in opt.last_gain:
            continue
        sig, g = opt.last_gain[id(p)]
        g = g[torch.argsort(sig, descending=True)]
        q += torch.stack([c.mean() for c in g.chunk(4)])
        k += 1
    return (q / k).tolist() if k else None


def main(variant, batch, lr, steps, seed, n_eval=10):
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
    if variant == "muon":
        mopt = torch.optim.Muon(mu, lr=lr, weight_decay=0.0)
    else:
        mopt = CVMuon(mu, lr=lr, gate=variant)
    opts = [mopt, torch.optim.AdamW(rest, lr=3e-3, weight_decay=0.0)]
    base = [[gp["lr"] for gp in o.param_groups] for o in opts]

    log, t_train = [], 0.0
    every = max(1, steps // n_eval)
    for s in range(steps + 1):
        if s % every == 0 or s == steps:
            vl, acc = evaluate(model, val, loss_fn)
            e = {"step": s, "time": t_train, "val_loss": vl, "val_acc": acc}
            if variant != "muon" and s > 0:
                e["gain"] = gain_profile(mopt, mu)
            log.append(e)
        if s == steps:
            break
        t0 = time.perf_counter()
        fr = 1.0 - s / steps
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
    r = {"variant": variant, "batch": batch, "lr": lr, "steps": steps, "seed": seed,
         "ms_per_step": 1e3 * t_train / steps, "log": log}
    gs = [e["gain"] for e in log if e.get("gain")]
    gtxt = ""
    if gs:
        fmt = lambda q: "[" + " ".join(f"{v:.2f}" for v in q) + "]"
        gtxt = f" gain(начало/середина/конец)={fmt(gs[0])}{fmt(gs[len(gs) // 2])}{fmt(gs[-1])}"
    mid = log[len(log) // 2]["val_loss"]
    print(f"RESULT {variant} batch={batch} lr={lr} seed={seed} steps={steps} ms/step={r['ms_per_step']:.1f} "
          f"val_loss(середина)={mid:.4f} val_loss={log[-1]['val_loss']:.4f}{gtxt}", flush=True)
    return r


if __name__ == "__main__":
    a = sys.argv
    r = main(a[1], int(a[2]), float(a[3]), int(a[4]), int(a[5]))
    if r["steps"] >= 400:
        with open("results/cv_runs.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")
