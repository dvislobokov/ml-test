"""CV-Muon: Muon с перекрёстной проверкой направлений (собственная идея этого исследования).

Проблема обычного Muon. Обновление — полярный множитель импульса polar(M) = U·Vᵀ:
все сингулярные направления получают единичный шаг, в том числе те, где в импульсе
один шум. По одной матрице M отличить сигнал от шума нельзя.

Идея. Импульс — сумма градиентов ~40 независимых батчей. Разложим его на два потока:
градиенты чётных шагов и градиенты нечётных, M = M₁ + M₂. Сигнал в потоках общий,
шум независим, поэтому согласие потоков вдоль направления измеряет долю сигнала в нём,
и шаг вдоль направления можно умножить на эту долю. Лишнего обратного прохода нет.

Варианты (gate):
  exact — контроль: точный полярный множитель без фильтра (через собственное разложение);
  v1    — направления (U, V) из полного импульса M; коэффициент направления i —
          согласованность потоков γ = 2·p·q / (|p|² + |q|²), p = M₁v, q = M₂v.
          Оценка смещена: направления выбраны по сумме обоих потоков;
  v2    — честная перекрёстная проверка: направления из одного потока, коэффициент —
          проекция другого потока на них (uᵀM₂v / σ); делается в обе стороны и усредняется.
Малые размеры (n×n) считаются в float64 через eigh: это проверка идеи, а не скорости.
"""
import torch


def _eig(M):
    """Собственное разложение MᵀM. Возвращает V [n,n] (float32) и σ [n] по возрастанию."""
    lam, V = torch.linalg.eigh((M.T @ M).double())
    return V.float(), lam.clamp_min(0).sqrt().float()


def _safe_div(g, sig, tol=1e-4):
    ok = sig > tol * sig.max()
    return torch.where(ok, g / sig.clamp_min(1e-30), torch.zeros_like(g))


class CVMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.03, momentum=0.95, gate="v2"):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.gate = gate
        self.last_gain = {}  # id(p) -> (σ, коэффициенты) для диагностики

    def _update(self, p, M, Mc, Mo, a, b):
        """M = Mc + Mo, все [m, n], m >= n; a, b — суммы весов потоков."""
        if self.gate == "exact" or b == 0.0:
            V, sig = _eig(M)
            g = torch.ones_like(sig)
            self.last_gain[id(p)] = (sig, g)
            return (M @ V) @ (_safe_div(g, sig, 1e-3)[:, None] * V.T)
        if self.gate == "v1":
            V, sig = _eig(M)
            P, Q = Mc @ V, Mo @ V
            gam = 2 * (P * Q).sum(0) / ((b / a) * (P * P).sum(0) + (a / b) * (Q * Q).sum(0) + 1e-30)
            gam = gam.clamp(0, 1)
            g = 2 * gam / (1 + gam)  # согласованность половин -> доля сигнала в сумме
            self.last_gain[id(p)] = (sig, g)
            return (P + Q) @ (_safe_div(g, sig)[:, None] * V.T)
        # v2: направления из одного потока, проверка другим, в обе стороны
        X, gs = 0.0, []
        for Mh, Mx, wh, wx in ((Mc, Mo, a, b), (Mo, Mc, b, a)):
            V, sig = _eig(Mh)
            P = Mh @ V  # = U·Σ
            t = (P * (Mx @ V)).sum(0)  # σ_i · u_iᵀ Mx v_i
            g = ((wh / wx) * _safe_div(t, sig) / sig.clamp_min(1e-30)).clamp(0, 1)
            g = torch.where(sig > 1e-4 * sig.max(), g, torch.zeros_like(g))
            X = X + 0.5 * (P @ (_safe_div(g, sig)[:, None] * V.T))
            gs.append((sig, g))
        self.last_gain[id(p)] = gs[0]
        return X

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            beta = grp["momentum"]
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["B"] = [torch.zeros_like(p), torch.zeros_like(p)]
                    st["s"] = [0.0, 0.0]  # суммы весов в потоках
                    st["t"] = 0
                B, s = st["B"], st["s"]
                c = st["t"] % 2  # поток, в который идёт текущий градиент
                o = 1 - c
                st["t"] += 1
                B[c].mul_(beta).add_(p.grad)
                B[o].mul_(beta)
                s[c] = beta * s[c] + 1.0
                s[o] = beta * s[o]
                # импульс Нестерова, разложенный на два потока: M = Mc + Mo
                Mc = p.grad.add(B[c], alpha=beta)
                Mo = B[o] * beta
                a, b = 1.0 + beta * s[c], beta * s[o]
                M = Mc + Mo
                tall = M.shape[0] >= M.shape[1]
                if not tall:
                    M, Mc, Mo = M.T, Mc.T, Mo.T
                X = self._update(p, M, Mc, Mo, a, b)
                if not tall:
                    X = X.T
                p.add_(X, alpha=-grp["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)
