"""Следящая ортогонализация (TrackMuon) — собственный алгоритм этого исследования.

Что делает Muon. Его обновление O = polar(M) — решение задачи Прокруста:
O = argmax ⟨O, M⟩ по матрицам с ортонормированными столбцами. Muon решает её с нуля
на каждом шаге (5 итераций Ньютона–Шульца, 15 матричных умножений) и для этого неявно
делит на сингулярные числа импульса. Малые сингулярные числа — в основном шум, и
деление на них было причиной всех наших неудач (тёплый старт, ленивый пересчёт).

Идея. Не решать задачу заново, а хранить ортогональное направление O как состояние
и на каждом шаге делать один шаг градиентного подъёма функции ⟨O, G_t⟩ по многообразию
ортогональных матриц:

    Y = O + α · G_t / ‖G_t‖          (толчок в сторону свежего градиента)
    O = Y · (1.5·I − 0.5·YᵀY)        (одна кубическая итерация возвращает ортогональность)
    W = W − lr · O

Свойства:
  * неподвижная точка при постоянном градиенте — ровно polar(G), то есть шаг Muon;
  * деления на сингулярные числа нет: O всегда почти ортогональна, итерация устойчива;
  * направление с относительной силой s поворачивается со скоростью ~α·s за шаг:
    сильные устойчивые направления подхватываются быстро, слабые — только если они
    держатся долго, а шум (меняющий знак от шага к шагу) не накапливается вовсе.
    Отдельный буфер импульса не нужен: память встроена в само состояние O;
  * цена — 2 больших матричных умножения на шаг вместо 10.

Параметры: alpha — скорость слежения (аналог 1−β импульса), beta — необязательное
предварительное усреднение градиента, iters — число кубических итераций за шаг.
"""
import torch

from fast_muon import cold_polar


class TrackMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.03, alpha=0.3, beta=0.0, iters=1, refresh=0, refresh_beta=0.95):
        super().__init__(params, dict(lr=lr))
        self.alpha, self.beta, self.iters = alpha, beta, iters
        # refresh=K: раз в K шагов направление пересчитывается «по-честному», как в Muon
        # (polar от импульса Нестерова), а между пересчётами — слежение (alpha>0) или
        # просто повторное использование прошлого направления (alpha=0).
        self.refresh, self.refresh_beta = refresh, refresh_beta

    @torch.no_grad()
    def step(self):
        a = self.alpha
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                tall = p.shape[0] >= p.shape[1]
                G = p.grad if tall else p.grad.T
                if self.beta > 0:
                    if "buf" not in st:
                        st["buf"] = torch.zeros_like(G)
                    G = st["buf"].mul_(self.beta).add_(G)
                t = st["t"] = st.get("t", -1) + 1
                if self.refresh:
                    if "mom" not in st:
                        st["mom"] = torch.zeros_like(G)
                    mom = st["mom"].mul_(self.refresh_beta).add_(p.grad if tall else p.grad.T)
                if self.refresh and t % self.refresh == 0:
                    Gc = p.grad if tall else p.grad.T
                    O = cold_polar(Gc.add(mom, alpha=self.refresh_beta), 5)[0]
                elif "O" not in st:
                    O = cold_polar(G, 5)[0]  # единственный «холодный» расчёт — на первом шаге
                elif a == 0.0:
                    O = st["O"]
                else:
                    # деление на (1 + α/2) держит сингулярные числа Y в зоне сходимости кубической итерации
                    Y = st["O"].add(G, alpha=a / (float(G.norm()) + 1e-12)).div_(1.0 + 0.5 * a)
                    for _ in range(self.iters):
                        Y = torch.addmm(Y, Y, Y.T @ Y, beta=1.5, alpha=-0.5)
                    O = Y
                st["O"] = O
                p.add_(O if tall else O.T, alpha=-grp["lr"] * max(1.0, p.shape[0] / p.shape[1]) ** 0.5)
