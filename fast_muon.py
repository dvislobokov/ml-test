"""Muon с удешевлённой ортогонализацией.

Обычный Muon на каждом шаге считает приближённый полярный множитель импульса M
с нуля: 5 итераций Ньютона–Шульца, по 3 матричных умножения каждая (15 умножений).

Идея: полярный множитель можно записать как Q = M·S, где S ≈ (MᵀM)^(-1/2) —
матрица размера меньшей стороны. Импульс между шагами меняется мало, значит и S
меняется мало. Поэтому S можно переносить с прошлого шага:

  warm   : X0 = M·S_prev (уже почти ортогональна) → одна кубическая итерация
           X = X(1.5·I − 0.5·XᵀX), та же поправка дописывается в S.
           4 умножения вместо 15. Если X0 далека от ортогональной
           (‖X0ᵀX0 − I‖ велика) — честный холодный пересчёт.
  lazy-k : S честно пересчитывается раз в k шагов, между ними Q = M·S_prev
           (1 умножение).
  cold-n : обычный Muon с n итерациями (cold-5 — базовый вариант).
"""
import torch

NS = (3.4445, -4.7750, 2.0315)  # коэффициенты из оригинального Muon


def cold_polar(M, steps, want_S=False):
    """Ньютон–Шульц пятой степени с нуля. M: [m, n], m >= n. Возвращает X (и S: X = M·S)."""
    a, b, c = NS
    scale = 1.0 / (M.norm() + 1e-7)
    X = M * scale
    S = None
    for _ in range(steps):
        A = X.T @ X
        P = b * A + c * (A @ A)
        P.diagonal().add_(a)  # P = aI + bA + cA²
        X = X @ P
        if want_S:
            S = P if S is None else S @ P
    if want_S:
        S = S * scale
    return X, S


class FastMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, mode="cold", ns_steps=5,
                 lazy_k=4, restart_tol=0.5, dtype=torch.bfloat16):
        super().__init__(params, dict(lr=lr, momentum=momentum))
        self.mode, self.ns_steps, self.lazy_k, self.tol = mode, ns_steps, lazy_k, restart_tol
        self.dtype = dtype
        self.stats = {"cold": 0, "warm": 0, "lazy": 0}
        self.last = {}  # id(p) -> последнее применённое обновление (для диагностики)

    def _orth(self, M, st):
        """M: [m, n], m >= n. Считаем в bfloat16, как встроенный torch.optim.Muon:
        на процессорах с AMX это в ~4 раза быстрее float32 при том же качестве."""
        M = M.to(self.dtype)
        if self.mode == "cold":
            self.stats["cold"] += 1
            return cold_polar(M, self.ns_steps)[0]
        S = st.get("S")
        if self.mode == "lazy":
            if S is None or st["t"] % self.lazy_k == 0:
                self.stats["cold"] += 1
                X, st["S"] = cold_polar(M, self.ns_steps, want_S=True)
                return X
            self.stats["lazy"] += 1
            return M @ S
        # warm
        if S is not None:
            X = M @ S
            A = X.T @ X
            n = A.shape[0]
            # ‖A − I‖_F² / n = (‖A‖² − 2·tr A + n) / n — средний квадрат отклонения
            dev = ((A * A).sum() - 2 * A.diagonal().sum() + n) / n
            if dev < self.tol ** 2:
                self.stats["warm"] += 1
                P = -0.5 * A
                P.diagonal().add_(1.5)
                st["S"] = S @ P
                return X @ P
        self.stats["cold"] += 1
        X, st["S"] = cold_polar(M, self.ns_steps, want_S=True)
        return X

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["buf"] = torch.zeros_like(p)
                    st["t"] = 0
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                M = p.grad.add(buf, alpha=g["momentum"])  # Нестеров
                tall = M.shape[0] >= M.shape[1]
                X = self._orth(M if tall else M.T, st)
                X = X.to(p.dtype)
                if not tall:
                    X = X.T
                st["t"] += 1
                self.last[id(p)] = (M, X)
                p.add_(X, alpha=-g["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)


def polar_quality(M, X):
    """Косинус между X и точным полярным множителем M (через SVD) и разброс сингулярных чисел X."""
    U, _, Vh = torch.linalg.svd(M.double(), full_matrices=False)
    Q = U @ Vh
    Xd = X.double()
    cos = ((Q * Xd).sum() / (Q.norm() * Xd.norm())).item()
    sv = torch.linalg.svdvals(Xd)
    return cos, sv.min().item(), sv.max().item()
