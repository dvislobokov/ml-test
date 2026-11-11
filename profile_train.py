"""Эксперимент 1: куда уходит время шага обучения (torch.profiler, CPU).

Для каждой модели профилируется N шагов обучения. Время раскладывается:
  * по фазам шага  — данные / прямой проход / loss / обратный проход / оптимизатор;
  * по операциям   — self CPU time каждой aten-операции, сгруппированное в категории.
"""
import json
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from common import MLP, MiniGPT, MiniResNet, load_mnist, load_text, n_params, text_batch

PHASES = ["1_data", "2_forward", "3_loss", "4_backward", "5_optimizer"]

# категория -> подстроки имён операций (проверяются по порядку)
CATEGORIES = [
    ("свёртки", ["conv", "mkldnn"]),
    ("матричные умножения", ["addmm", "aten::mm", "bmm", "matmul", "aten::linear", "baddbmm", "einsum"]),
    ("внимание (softmax, маска)", ["softmax", "scaled_dot", "masked_fill", "tril", "where"]),
    ("нормализация", ["batch_norm", "layer_norm", "native_"]),
    ("активации", ["relu", "gelu", "threshold", "clamp"]),
    ("loss (log_softmax, nll)", ["log_softmax", "nll_loss", "cross_entropy"]),
    ("оптимизатор (поэлементно)", ["addcdiv", "addcmul", "lerp", "sqrt", "aten::mul_", "aten::add_", "aten::div", "zero_", "aten::mul", "Optimizer"]),
    ("эмбеддинги/индексация", ["embedding", "index", "gather", "scatter", "aten::stack", "aten::cat", "select", "slice"]),
    ("копирование/раскладка памяти", ["copy", "contiguous", "clone", "aten::to", "reshape", "view", "permute", "transpose", "expand", "aten::t", "empty", "zeros", "ones", "fill", "resize", "randint", "randperm", "arange"]),
    ("поэлементная арифметика/суммы", ["aten::add", "aten::sub", "aten::sum", "aten::mean", "aten::neg", "aten::div", "aten::pow"]),
]


def categorize(name):
    low = name
    for cat, keys in CATEGORIES:
        if any(k in low for k in keys):
            return cat
    return "прочее (autograd, диспетчер)"


def make_task(kind):
    g = torch.Generator().manual_seed(0)
    if kind == "mlp" or kind == "cnn":
        xtr, ytr, _, _ = load_mnist()
        model = MLP() if kind == "mlp" else MiniResNet()
        bs = 128

        def batch():
            ix = torch.randint(len(xtr), (bs,), generator=g)
            return xtr[ix], ytr[ix]

        loss_fn = lambda out, y: F.cross_entropy(out, y)
    else:
        tr, _, vocab = load_text()
        model = MiniGPT(vocab)
        batch = lambda: text_batch(tr, 32, 128, g)
        loss_fn = lambda out, y: F.cross_entropy(out.view(-1, out.shape[-1]), y.view(-1))
    return model, batch, loss_fn


def step(model, opt, batch, loss_fn):
    with record_function("1_data"):
        x, y = batch()
    with record_function("2_forward"):
        out = model(x)
    with record_function("3_loss"):
        loss = loss_fn(out, y)
    with record_function("4_backward"):
        opt.zero_grad(set_to_none=True)
        loss.backward()
    with record_function("5_optimizer"):
        opt.step()


def run(kind, steps=60, warmup=10):
    torch.manual_seed(0)
    model, batch, loss_fn = make_task(kind)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(warmup):
        step(model, opt, batch, loss_fn)

    # чистое время шага без профайлера — чтобы оценить его накладные расходы
    t0 = time.perf_counter()
    for _ in range(steps):
        step(model, opt, batch, loss_fn)
    clean_ms = (time.perf_counter() - t0) / steps * 1e3

    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(steps):
            step(model, opt, batch, loss_fn)
    ev = prof.key_averages()

    phases = {e.key: e.cpu_time_total / steps / 1e3 for e in ev if e.key in PHASES}
    ops, cats = {}, defaultdict(float)
    for e in ev:
        if e.key in PHASES or e.self_cpu_time_total <= 0:
            continue
        ms = e.self_cpu_time_total / steps / 1e3
        ops[e.key] = ms
        cats[categorize(e.key)] += ms
    total_self = sum(ops.values())
    return {
        "model": kind,
        "params": n_params(model),
        "clean_step_ms": clean_ms,
        "profiled_step_ms": sum(phases.values()),
        "phases_ms": phases,
        "categories_ms": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "top_ops_ms": dict(sorted(ops.items(), key=lambda kv: -kv[1])[:15]),
        "total_self_ms": total_self,
    }


def show(r):
    print(f"\n=== {r['model']}: {r['params']/1e6:.2f}M параметров, "
          f"шаг {r['clean_step_ms']:.1f} мс (под профайлером {r['profiled_step_ms']:.1f} мс)")
    tot = sum(r["phases_ms"].values())
    print("  Фазы шага:")
    for p in PHASES:
        v = r["phases_ms"].get(p, 0)
        print(f"    {p:<14}{v:8.2f} мс {100*v/tot:5.1f}%")
    print("  Категории операций (self CPU time):")
    for c, v in r["categories_ms"].items():
        print(f"    {c:<34}{v:8.2f} мс {100*v/r['total_self_ms']:5.1f}%")
    print("  Топ операций:")
    for o, v in list(r["top_ops_ms"].items())[:10]:
        print(f"    {o:<46}{v:8.2f} мс {100*v/r['total_self_ms']:5.1f}%")


if __name__ == "__main__":
    kinds = sys.argv[1:] or ["mlp", "cnn", "gpt"]
    print(f"torch {torch.__version__}, потоков: {torch.get_num_threads()}")
    res = [run(k) for k in kinds]
    for r in res:
        show(r)
    if len(kinds) == 3:
        with open("results/profile.json", "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
