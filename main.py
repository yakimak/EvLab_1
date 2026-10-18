"""
Лабораторная работа №1.
Вариант 4. Функция Экли, размерность d=9, область поиска [-32.768, 32.768].
Студент: Якимов Максим Андреевич
Группа: Б24-517, номер в группе 30.

Запуск:
    python main.py
Результаты:
    - lab1_out/lab1_results.csv
    - lab1_out/lab1_random.csv
    - lab1_out/lab1_convergence.png
    - lab1_out/lab1_comparison.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
from typing import Callable, List, Tuple

# ---------- Настройки воспроизводимости ----------
BASE_SEED = 42
N_RUNS = 20          # число независимых запусков
DIM = 9              # d = 5 + (V mod 6) = 5 + (4 mod 6) = 9
LOWER = -32.768
UPPER = 32.768

# Глобальный минимум для функции Экли
REFERENCE_BEST = 0.0


# ---------- Целевая функция ----------
def ackley(x: np.ndarray) -> float:
    """Функция Экли: f(x) = -20*exp(-0.2*sqrt(1/d * sum(x_i^2))) - exp(1/d * sum(cos(2*pi*x_i))) + 20 + e."""
    d = len(x)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / d))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * x)) / d)
    return float(term1 + term2 + 20.0 + np.e)


def ackley_batch(X: np.ndarray) -> np.ndarray:
    """Векторная версия для популяции."""
    d = X.shape[1]
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.sum(X**2, axis=1) / d))
    term2 = -np.exp(np.sum(np.cos(2.0 * np.pi * X), axis=1) / d)
    return term1 + term2 + 20.0 + np.e


# ---------- Генетический алгоритм ----------
@dataclass
class GAConfig:
    pop_size: int = 100
    max_gens: int = 300
    tournament_k: int = 3
    p_crossover: float = 0.9
    p_mutation: float = 0.2
    sigma: float = 0.15          # масштаб гауссовой мутации (доля от диапазона)
    eta_sbx: float = 15.0        # параметр SBX-кроссовера
    elite_count: int = 2
    seed: int = BASE_SEED
    tag: str = "default"


def init_population(cfg: GAConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    return rng.uniform(LOWER, UPPER, size=(cfg.pop_size, DIM))


def tournament_selection(fitness: np.ndarray, k: int, rng: np.random.Generator) -> int:
    idx = rng.integers(0, len(fitness), size=k)
    # минимизация
    return idx[int(np.argmin(fitness[idx]))]


def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta: float,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """SBX-кроссовер для вещественных векторов."""
    u = rng.random(size=p1.shape)
    mask = u <= 0.5
    beta = np.where(mask,
                    (2.0 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0)))
    c1 = 0.5 * ((p1 + p2) + beta * np.abs(p2 - p1))
    c2 = 0.5 * ((p1 + p2) - beta * np.abs(p2 - p1))
    return c1, c2


def gaussian_mutation(x: np.ndarray, sigma: float,
                      rng: np.random.Generator) -> np.ndarray:
    y = x + rng.normal(0.0, sigma * (UPPER - LOWER), size=x.shape)
    # отражение от границ
    y = np.clip(y, LOWER, UPPER)
    return y


def run_ga(cfg: GAConfig) -> dict:
    rng = np.random.default_rng(cfg.seed)
    pop = init_population(cfg)
    fit = ackley_batch(pop)

    best_idx = int(np.argmin(fit))
    history_best = [fit[best_idx]]
    history_mean = [float(fit.mean())]
    elite_archive = [(fit[best_idx], pop[best_idx].copy())]

    evals = cfg.pop_size
    for gen in range(1, cfg.max_gens + 1):
        new_pop = []
        # элитизм
        order = np.argsort(fit)
        for i in range(cfg.elite_count):
            new_pop.append(pop[order[i]].copy())

        while len(new_pop) < cfg.pop_size:
            i1 = tournament_selection(fit, cfg.tournament_k, rng)
            i2 = tournament_selection(fit, cfg.tournament_k, rng)
            c1, c2 = sbx_crossover(pop[i1], pop[i2], cfg.eta_sbx, rng) \
                if rng.random() < cfg.p_crossover else (pop[i1].copy(), pop[i2].copy())
            c1 = gaussian_mutation(c1, cfg.sigma, rng) if rng.random() < cfg.p_mutation else c1
            c2 = gaussian_mutation(c2, cfg.sigma, rng) if rng.random() < cfg.p_mutation else c2
            new_pop.append(c1)
            if len(new_pop) < cfg.pop_size:
                new_pop.append(c2)

        pop = np.array(new_pop)
        fit = ackley_batch(pop)
        evals += cfg.pop_size

        gen_best_idx = int(np.argmin(fit))
        history_best.append(fit[gen_best_idx])
        history_mean.append(float(fit.mean()))
        if fit[gen_best_idx] < elite_archive[0][0]:
            elite_archive = [(fit[gen_best_idx], pop[gen_best_idx].copy())]

    return {
        "best": elite_archive[0][0],
        "best_x": elite_archive[0][1],
        "history_best": history_best,
        "history_mean": history_mean,
        "evals": evals,
    }


def random_search(budget: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.uniform(LOWER, UPPER, size=(budget, DIM))
    F = ackley_batch(X)
    best_idx = int(np.argmin(F))
    return {"best": F[best_idx], "best_x": X[best_idx]}


# ---------- Эксперименты ----------
def experiment_series(configs: List[GAConfig], n_runs: int = N_RUNS) -> dict:
    results = {}
    for cfg in configs:
        runs_best = []
        runs_history_best = []
        runs_history_mean = []
        for r in range(n_runs):
            c = GAConfig(**{**cfg.__dict__, "seed": cfg.seed + r, "tag": cfg.tag})
            res = run_ga(c)
            runs_best.append(res["best"])
            runs_history_best.append(res["history_best"])
            runs_history_mean.append(res["history_mean"])
        results[cfg.tag] = {
            "bests": np.array(runs_best),
            "hist_best": np.array(runs_history_best),
            "hist_mean": np.array(runs_history_mean),
            "cfg": cfg,
        }
    return results


def save_results_csv(results: dict, path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "run", "best_f", "mean_f", "std_f", "median_f", "worst_f"])
        for tag, data in results.items():
            b = data["bests"]
            w.writerow([tag, "summary",
                        float(b.min()), float(b.mean()), float(b.std()),
                        float(np.median(b)), float(b.max())])
            for i, v in enumerate(b):
                w.writerow([tag, i, float(v), "", "", "", ""])


def save_random_csv(bests: np.ndarray, path: str):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "run", "best_f", "mean_f", "std_f", "median_f", "worst_f"])
        w.writerow(["random_search", "summary",
                    float(bests.min()), float(bests.mean()), float(bests.std()),
                    float(np.median(bests)), float(bests.max())])
        for i, v in enumerate(bests):
            w.writerow(["random_search", i, float(v), "", "", "", ""])


def plot_convergence(results: dict, path: str):
    plt.figure(figsize=(9, 5))
    for tag, data in results.items():
        hb = data["hist_best"]
        mean_curve = hb.mean(axis=0)
        std_curve = hb.std(axis=0)
        gens = np.arange(len(mean_curve))
        plt.plot(gens, mean_curve, label=f"{tag} (mean)")
        plt.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
    plt.axhline(REFERENCE_BEST, color="k", linestyle="--", linewidth=1,
                label=f"reference best = {REFERENCE_BEST}")
    plt.xlabel("Поколение")
    plt.ylabel("Лучшее значение f(x)")
    plt.title("Сходимость ГА на функции Экли (d=9)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison(results: dict, path: str):
    plt.figure(figsize=(8, 5))
    tags = list(results.keys())
    data = [results[t]["bests"] for t in tags]
    bp = plt.boxplot(data, patch_artist=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for patch, color in zip(bp["boxes"], colors[:len(tags)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.xticks(range(1, len(tags) + 1), tags, rotation=15, ha="right")
    plt.axhline(REFERENCE_BEST, color="k", linestyle="--", label="reference best")
    plt.ylabel("Лучшее f(x) за запуск")
    plt.title("Распределение результатов по 20 запускам (Функция Экли)")
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    os.makedirs("lab1_out", exist_ok=True)

    # --- Конфигурации для сравнения (меняем ОДИН фактор) ---
    cfgs = [
        GAConfig(pop_size=100, max_gens=300, p_mutation=0.20, sigma=0.15, tag="base"),
        GAConfig(pop_size=200, max_gens=300, p_mutation=0.20, sigma=0.15, tag="pop=200"),
        GAConfig(pop_size=100, max_gens=300, p_mutation=0.05, sigma=0.15, tag="p_mut=0.05"),
        GAConfig(pop_size=100, max_gens=300, p_mutation=0.50, sigma=0.15, tag="p_mut=0.50"),
    ]

    print("Запуск серии экспериментов ГА...")
    ga_results = experiment_series(cfgs, n_runs=N_RUNS)
    save_results_csv(ga_results, "lab1_out/lab1_results.csv")
    plot_convergence(ga_results, "lab1_out/lab1_convergence.png")
    plot_comparison(ga_results, "lab1_out/lab1_comparison.png")

    # --- Сравнение со случайным поиском ---
    budget = cfgs[0].pop_size * cfgs[0].max_gens  # равный бюджет оценок
    rs_bests = []
    for r in range(N_RUNS):
        rs = random_search(budget=budget, seed=BASE_SEED + 1000 + r)
        rs_bests.append(rs["best"])
    rs_bests = np.array(rs_bests)
    save_random_csv(rs_bests, "lab1_out/lab1_random.csv")

    # --- Итоговая сводка ---
    print("\n=== Сводка по 20 запускам ===")
    print(f"{'конфиг':<14} {'min':>10} {'mean':>10} {'median':>10} {'std':>8} {'max':>10}")
    for tag, data in ga_results.items():
        b = data["bests"]
        print(f"{tag:<14} {b.min():10.4f} {b.mean():10.4f} "
              f"{np.median(b):10.4f} {b.std():8.4f} {b.max():10.4f}")
    print(f"{'random_search':<14} {rs_bests.min():10.4f} {rs_bests.mean():10.4f} "
          f"{np.median(rs_bests):10.4f} {rs_bests.std():8.4f} {rs_bests.max():10.4f}")
    print(f"\nОриентир глобального минимума для d=9: {REFERENCE_BEST}")
    print("\nФайлы сохранены в папку lab1_out/")


if __name__ == "__main__":
    main()