"""Диагностика к эксперименту 6: сколько направлений обновления Muon — сигнал, а сколько шум.

Обучаем мини-GPT встроенным Muon, параллельно ведём два потока импульса (чётные и нечётные
шаги). В контрольной точке считаем «истинный» градиент G* на большом батче и для каждого
сингулярного направления (u_i, v_i) импульса M сравниваем:

  truth — cos(u_i, G* v_i): согласие направления с истинным градиентом (1 — чистый сигнал,
          0 — шум, <0 — шаг в неверную сторону);
  est   — оценка того же без G*: cos(M₁ v_i, M₂ v_i), согласие двух потоков импульса.

  python3 cv_diag.py 8 300        # батч, шаг контрольной точки
"""
import sys

import torch
import torch.nn.functional as F

from common import MiniGPT, load_text, text_batch

IDX = [0, 1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 127]


def main(batch, at_step, lr=0.03, beta=0.95, true_seqs=2048):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(0)
    tr, _, vocab = load_text()
    model = MiniGPT(vocab)
    loss_fn = lambda out, y: F.cross_entropy(out.view(-1, out.shape[-1]), y.view(-1))
    named = list(model.named_parameters())
    is_mu = lambda n, p: p.ndim == 2 and n.startswith("blocks.")
    mu = {n: p for n, p in named if is_mu(n, p)}
    opts = [torch.optim.Muon(list(mu.values()), lr=lr, weight_decay=0.0),
            torch.optim.AdamW([p for n, p in named if not is_mu(n, p)], lr=3e-3, weight_decay=0.0)]
    B = {n: [torch.zeros_like(p), torch.zeros_like(p)] for n, p in mu.items()}

    for s in range(at_step):
        x, y = text_batch(tr, batch, 128, g)
        loss = loss_fn(model(x), y)
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        c = s % 2
        with torch.no_grad():
            for n, p in mu.items():
                B[n][c].mul_(beta).add_(p.grad)
                B[n][1 - c].mul_(beta)
        if s < at_step - 1:
            for o in opts:
                o.step()

    # потоки импульса Нестерова на последнем шаге (до применения обновления)
    c = (at_step - 1) % 2
    Ms = {n: (p.grad + beta * B[n][c], beta * B[n][1 - c]) for n, p in mu.items()}

    # «истинный» градиент в той же точке
    Gt = {n: torch.zeros_like(p) for n, p in mu.items()}
    k = true_seqs // 32
    for _ in range(k):
        x, y = text_batch(tr, 32, 128, g)
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        for n, p in mu.items():
            Gt[n] += p.grad / k

    print(f"батч {batch}, шаг {at_step}; номера направлений по убыванию сингулярного числа")
    print(f"{'слой':<16}{'':<7}" + "".join(f"{i:>6}" for i in IDX) + "   среднее  доля<0")
    tot_t, tot_e = [], []
    for n in mu:
        Mc, Mo = Ms[n]
        G = Gt[n]
        if Mc.shape[0] < Mc.shape[1]:
            Mc, Mo, G = Mc.T, Mo.T, G.T
        M = (Mc + Mo).double()
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        V = Vh.T
        GV = G.double() @ V
        truth = (U * GV).sum(0) / GV.norm(dim=0).clamp_min(1e-30)
        P, Q = Mc.double() @ V, Mo.double() @ V
        est = (P * Q).sum(0) / (P.norm(dim=0) * Q.norm(dim=0)).clamp_min(1e-30)
        tot_t.append(truth)
        tot_e.append(est)
        if n.startswith("blocks.1."):
            for name, v in (("truth", truth), ("est", est), ("σ/σmax", S / S[0])):
                tail = f"   {v.mean():7.3f}  {(v < 0).double().mean():6.2f}" if name != "σ/σmax" else ""
                print(f"{n[7:]:<16}{name:<7}" + "".join(f"{v[i]:6.2f}" for i in IDX) + tail)
    T, E = torch.cat(tot_t), torch.cat(tot_e)
    corr = torch.corrcoef(torch.stack([T, E]))[0, 1].item()
    print(f"\nвсе {len(tot_t)} матриц, {len(T)} направлений:")
    print(f"  согласие с истинным градиентом: среднее {T.mean():.3f}; направлений с truth > 0,5: {(T > 0.5).double().mean():.1%}; "
          f"с truth < 0,1: {(T < 0.1).double().mean():.1%}; в неверную сторону: {(T < 0).double().mean():.1%}")
    print(f"  корреляция оценки по двум потокам с истиной: {corr:.3f}")
    # первый порядок: какую долю «пользы» шага дают лучшие по оценке направления
    for name, key in (("по оценке est", E), ("по истине (предел)", T)):
        order = torch.argsort(key, descending=True)
        # вклад направления в убывание потерь ∝ u_iᵀ G* v_i; здесь берём truth как прокси знака и силы
        cum = torch.cumsum(T[order], 0) / T.sum()
        ks = [int(len(T) * f) for f in (0.1, 0.25, 0.5)]
        print(f"  отбор {name}: лучшие 10% / 25% / 50% направлений дают "
              + " / ".join(f"{cum[k - 1]:.0%}" for k in ks) + " суммарного согласия")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
