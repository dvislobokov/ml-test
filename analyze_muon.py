"""Сводка по results/muon_runs.jsonl: таблица и график «потери от времени»."""
import json
from collections import defaultdict
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

runs = [json.loads(l) for l in open("results/muon_runs.jsonl")]

# Время шага на общей машине шумит, поэтому для оси времени берём медиану по
# всем прогонам данной пары (задача, оптимизатор) и умножаем на номер шага.
ms = defaultdict(list)
for r in runs:
    ms[(r["task"], r["opt"])].append(r["ms_per_step"])
ms = {k: median(v) for k, v in ms.items()}

print(f"{'задача':<5}{'опт.':<7}{'lr':>8}{'шагов':>7}{'seed':>5}{'мс/шаг':>8}{'опт. мс':>9}{'val loss':>10}{'acc':>8}")
for r in sorted(runs, key=lambda r: (r["task"], r["steps"], r["opt"], r["lr"], r["seed"])):
    e = r["log"][-1]
    print(f"{r['task']:<5}{r['opt']:<7}{r['lr']:>8g}{r['steps']:>7}{r['seed']:>5}"
          f"{r['ms_per_step']:>8.1f}{r['opt_ms_per_step']:>9.2f}{e['val_loss']:>10.4f}{e['val_acc']:>8.4f}")
print("\nмедианное время шага, мс:", {f"{k[0]}/{k[1]}": round(v, 1) for k, v in ms.items()})


def first_reach(r, target):
    for e in r["log"]:
        if e["val_loss"] <= target:
            return e["step"]
    return None


long = [r for r in runs if r["task"] == "gpt" and r["steps"] >= 1500]
if long:
    print("\nGPT, длинные прогоны: шаг и время достижения порога потерь на валидации")
    for target in (2.0, 1.8, 1.7, 1.6):
        for r in sorted(long, key=lambda r: (r["opt"], r["seed"])):
            s = first_reach(r, target)
            t = "не достигнут" if s is None else f"шаг {s:5d}, {s * ms[('gpt', r['opt'])] / 1e3:6.1f} с"
            print(f"  порог {target}: {r['opt']:<6} seed {r['seed']}: {t}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    col = {"adamw": "#4c78a8", "muon": "#e45756"}
    for r in long:
        steps = [e["step"] for e in r["log"]]
        loss = [e["val_loss"] for e in r["log"]]
        k = ("gpt", r["opt"])
        lab = f"{r['opt']} lr={r['lr']:g}" if r["seed"] == long[0]["seed"] else None
        ax[0].plot(steps, loss, color=col[r["opt"]], label=lab)
        ax[1].plot([s * ms[k] / 1e3 for s in steps], loss, color=col[r["opt"]], label=lab)
    ax[0].set_xlabel("шаг обучения")
    ax[1].set_xlabel("время обучения, с")
    for a in ax:
        a.set_ylabel("потери на валидации")
        a.set_ylim(1.4, 2.8)
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Мини-GPT: Muon против AdamW (2 seed)")
    fig.tight_layout()
    fig.savefig("results/muon_vs_adamw_gpt.png", dpi=130)
    print("\nграфик: results/muon_vs_adamw_gpt.png")
