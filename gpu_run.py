"""Запуск сравнения оптимизаторов на GPU (или на CPU, если GPU нет).

Модель по умолчанию — посимвольный GPT на ~10,7 млн параметров (6 слоёв, d=384, контекст 256),
та же постановка, что nanoGPT shakespeare_char. Все варианты получают одинаковый бюджет времени.

  ./fetch_data.sh
  python3 gpu_run.py adamw 120      # AdamW, 120 секунд обучения
  python3 gpu_run.py muon  120      # встроенный torch.optim.Muon
  python3 gpu_run.py muon3 120      # он же с 3 итерациями
  python3 gpu_run.py track 120      # гибрид: пересчёт раз в 4 шага + слежение (track_muon.py)

Необязательные параметры: --batch 64 --d 384 --layers 6 --heads 6 --ctx 256 --lr <шаг> --seed 0
                          --refresh 4 --alpha 1.0 --no-amp --threads N
Результат дописывается в results/gpu_runs.jsonl.
"""
import argparse
import json
import time

import torch
import torch.nn.functional as F

from common import MiniGPT, load_text, n_params
from track_muon import TrackMuon

ap = argparse.ArgumentParser()
ap.add_argument("variant", choices=["adamw", "muon", "muon3", "track"])
ap.add_argument("budget", type=float)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--d", type=int, default=384)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--heads", type=int, default=6)
ap.add_argument("--ctx", type=int, default=256)
ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--refresh", type=int, default=4)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--no-amp", action="store_true")
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()

if a.threads:
    torch.set_num_threads(a.threads)
dev = "cuda" if torch.cuda.is_available() else "cpu"
amp = dev == "cuda" and not a.no_amp
torch.manual_seed(a.seed)
if dev == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

tr, va, vocab = load_text()
tr, va = tr.to(dev), va.to(dev)
offs = torch.arange(a.ctx, device=dev)


def get_batch(ids, n, gen=None):
    ix = torch.randint(len(ids) - a.ctx - 1, (n, 1), device=dev, generator=gen)
    return ids[ix + offs], ids[ix + offs + 1]


model = MiniGPT(vocab, d=a.d, layers=a.layers, heads=a.heads, ctx=a.ctx).to(dev)
vgen = torch.Generator(device=dev).manual_seed(123)
val = [get_batch(va, 64, vgen) for _ in range(10)]
loss_fn = lambda out, y: F.cross_entropy(out.view(-1, out.shape[-1]).float(), y.view(-1))

named = list(model.named_parameters())
is_mu = lambda n, p: p.ndim == 2 and n.startswith("blocks.")
mu = [p for n, p in named if is_mu(n, p)]
rest = [p for n, p in named if not is_mu(n, p)]
if a.variant == "adamw":
    opts = [torch.optim.AdamW(model.parameters(), lr=a.lr or 1e-3, weight_decay=0.0)]
else:
    lr = a.lr or 0.03
    if a.variant == "track":
        mopt = TrackMuon(mu, lr=lr, alpha=a.alpha, refresh=a.refresh)
    else:
        mopt = torch.optim.Muon(mu, lr=lr, weight_decay=0.0, ns_steps=3 if a.variant == "muon3" else 5)
    opts = [mopt, torch.optim.AdamW(rest, lr=3e-3, weight_decay=0.0)]
base = [[gp["lr"] for gp in o.param_groups] for o in opts]


def now():
    if dev == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def evaluate():
    model.eval()
    tot = 0.0
    for x, y in val:
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
            tot += loss_fn(model(x), y).item()
    model.train()
    return tot / len(val)


print(f"устройство: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "")
      + f", параметров {n_params(model)/1e6:.2f} млн, батч {a.batch}×{a.ctx}, bfloat16: {amp}", flush=True)

# прогрев (инициализация ядер, первый шаг) не входит в бюджет
x, y = get_batch(tr, a.batch)
with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
    loss_fn(model(x), y).backward()
model.zero_grad(set_to_none=True)

log, t_train, t_opt, steps, next_eval = [], 0.0, 0.0, 0, 0.0
every = max(2.0, a.budget / 20)
while True:
    done = t_train >= a.budget
    if t_train >= next_eval or done:
        vl = evaluate()
        log.append({"step": steps, "time": t_train, "val_loss": vl})
        print(f"{a.variant} t={t_train:6.1f}s шаг {steps:6d} val_loss={vl:.4f}", flush=True)
        next_eval += every
    if done:
        break
    t0 = now()
    fr = max(0.0, 1.0 - t_train / a.budget)
    for o, b in zip(opts, base):
        for gp, lr0 in zip(o.param_groups, b):
            gp["lr"] = lr0 * fr
    x, y = get_batch(tr, a.batch)
    with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
        loss = loss_fn(model(x), y)
    for o in opts:
        o.zero_grad(set_to_none=True)
    loss.backward()
    t1 = now()
    for o in opts:
        o.step()
    t2 = now()
    t_train += t2 - t0
    t_opt += t2 - t1
    steps += 1

r = {"variant": a.variant, "device": dev, "args": vars(a), "params": n_params(model), "steps": steps,
     "ms_per_step": 1e3 * t_train / steps, "opt_ms_per_step": 1e3 * t_opt / steps,
     "best_val": min(e["val_loss"] for e in log), "log": log}
print(f"RESULT {a.variant} {dev} шагов={steps} мс/шаг={r['ms_per_step']:.2f} оптимизатор={r['opt_ms_per_step']:.2f} мс "
      f"({100*t_opt/t_train:.0f}% шага) val_loss: финал {log[-1]['val_loss']:.4f}, лучший {r['best_val']:.4f}")
if a.budget >= 30:
    with open("results/gpu_runs.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
