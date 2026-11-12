"""Сводка по results/track_runs.jsonl (эксперимент 7)."""
import json
from collections import defaultdict
from statistics import mean

runs = [json.loads(l) for l in open("results/track_runs.jsonl")]


def name(r):
    if r["variant"] == "muon":
        return "Muon (5 итераций)"
    if r["variant"] == "muon3":
        return "Muon (3 итерации)"
    return f"гибрид, пересчёт раз в {r['refresh']}" if r["refresh"] else f"слежение, alpha={r['alpha']}"


def table(rs, title):
    if not rs:
        return
    print(f"\n{title}")
    g = defaultdict(list)
    for r in rs:
        # прогон, испорченный параллельным запуском: оптимизатор гибрида занял >15 мс вместо 7–10
        if r["variant"] == "track" and r["opt_ms_per_step"] > 15:
            continue
        g[(r["batch"], name(r))].append(r)
    print(f"{'батч':<6}{'вариант':<28}{'потери по seed':<26}{'среднее':>8}{'шагов':>8}{'мс/шаг':>8}{'опт., мс':>9}")
    for (b, n), v in sorted(g.items()):
        v.sort(key=lambda r: r["seed"])
        # при повторе одного seed берём последний прогон
        by_seed = {r["seed"]: r for r in v}
        v = list(by_seed.values())
        ls = [r["log"][-1]["val_loss"] for r in v]
        print(f"{b:<6}{n:<28}{' / '.join(f'{x:.3f}' for x in ls):<26}{mean(ls):>8.3f}"
              f"{mean(r['steps'] for r in v):>8.0f}{mean(r['ms_per_step'] for r in v):>8.1f}{mean(r['opt_ms_per_step'] for r in v):>9.1f}")


table([r for r in runs if not r.get("max_steps")], "Равный бюджет времени (45 с обучения)")
table([r for r in runs if r.get("max_steps")], "Равное число шагов (качество одного шага)")
