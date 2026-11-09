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
    with open(path, "w", newline="", encoding="utf-8") as f:
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
    with open(path, "w", newline="", encoding="utf-8") as f:
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


def save_histories(results: dict, path: str):
    """Сохраняет траектории сходимости для последующей перестройки графиков."""
    payload = {}
    for tag, data in results.items():
        payload[f"{tag}__best"] = data["hist_best"]
        payload[f"{tag}__mean"] = data["hist_mean"]
    np.savez_compressed(path, **payload)


def plot_flowchart(path: str):
    """Блок-схема генетического алгоритма (графическое представление)."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12.5, 11))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11.6)
    ax.axis("off")

    def box(x, y, w, h, text, color="#EAF2FB", edge="#2F5D8A", fs=10):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                           linewidth=1.6, edgecolor=edge, facecolor=color)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#12263A")

    def arrow(x1, y1, x2, y2, text="", color="#2F5D8A", fs=8.5, ha="left"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                            linewidth=1.6, color=color,
                            connectionstyle="arc3,rad=0.0")
        ax.add_patch(a)
        if text:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, text, fontsize=fs,
                    color=color, ha=ha, va="center")

    ax.text(7, 11.2, "Блок-схема генетического алгоритма (ЛР №1, функция Экли, d = 9)",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#12263A")

    cx, w = 4.0, 5.5
    steps = [
        ("Старт\n(seed = 42…61)", 10.3, "#EAF2FB", "#2F5D8A"),
        ("Инициализация популяции\nравномерно в [−32.768, 32.768]", 9.4, "#EAF2FB", "#2F5D8A"),
        ("Оценка фитнес-функции f(x)\n(функция Экли)", 8.5, "#EAF2FB", "#2F5D8A"),
        ("Элитизм: сохранить 2 лучшие особи", 7.3, "#EAF2FB", "#2F5D8A"),
        ("Турнирная селекция родителей (k = 3)", 6.4, "#EAF2FB", "#2F5D8A"),
        ("SBX-кроссовер (p = 0.9, η = 15)", 5.5, "#EAF2FB", "#2F5D8A"),
        ("Гауссова мутация\n(p_mut, σ = 0.15·(u−l))", 4.6, "#EAF2FB", "#2F5D8A"),
        ("Обработка выхода за границы\n(отражение / clip)", 3.7, "#EAF2FB", "#2F5D8A"),
        ("Новая популяция и оценка фитнеса", 2.8, "#FEF9E7", "#B7950B"),
        ("Достигнуто 300 поколений?", 1.7, "#EAF2FB", "#2F5D8A"),
        ("Вывод лучшего решения (x*, f(x*))\nи графиков", 0.45, "#E8F8F5", "#117A65"),
    ]
    for text, y, color, edge in steps:
        box(cx, y, w, 0.72, text, color=color, edge=edge)

    for i in range(len(steps) - 2):
        y_from = steps[i][1]
        y_to = steps[i + 1][1] + 0.72
        arrow(cx + w / 2, y_from, cx + w / 2, y_to)

    # Ветка «да»: остановка
    arrow(cx + w / 2, 1.7, cx + w / 2, 1.17, text="да", ha="left")

    # Ветка «нет»: возврат к элитизму по правой стороне
    loop_x = 11.8
    arrow(cx + w, 2.06, loop_x, 2.06, color="#B03A2E")
    arrow(loop_x, 2.06, loop_x, 7.66, color="#B03A2E")
    arrow(loop_x, 7.66, cx + w, 7.66, color="#B03A2E")
    ax.text(loop_x + 0.15, 4.9, "нет:\nследующее\nпоколение", fontsize=9,
            color="#B03A2E", ha="left", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_dashboard(results: dict, rs_bests: np.ndarray, path: str):
    """Сводный дашборд: сходимость, распределение, средние и таблица статистик."""
    from matplotlib.gridspec import GridSpec

    tags = list(results.keys())
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig = plt.figure(figsize=(17, 10.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30,
                  height_ratios=[1.05, 1.0])
    fig.suptitle("ЛР №1 · Генетический алгоритм на функции Экли (d = 9) · 20 независимых запусков",
                 fontsize=16, fontweight="bold", color="#12263A", y=0.98)

    ax1 = fig.add_subplot(gs[0, :2])
    for tag, color in zip(tags, colors):
        hb = results[tag]["hist_best"]
        mean_curve = hb.mean(axis=0)
        std_curve = hb.std(axis=0)
        gens = np.arange(len(mean_curve))
        ax1.plot(gens, np.maximum(mean_curve, 1e-5), color=color, label=tag, linewidth=2)
        ax1.fill_between(gens, np.maximum(mean_curve - std_curve, 1e-5),
                         np.maximum(mean_curve + std_curve, 1e-5), color=color, alpha=0.13)
    ax1.axhline(max(REFERENCE_BEST, 1e-5), color="k", linestyle="--", linewidth=1,
                label="глобальный минимум = 0")
    ax1.set_yscale("log")
    ax1.set_xlabel("Поколение")
    ax1.set_ylabel("Лучшее f(x) (log scale)")
    ax1.set_title("Кривые сходимости: среднее ± std по 20 запускам")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 2])
    data = [results[t]["bests"] for t in tags] + [rs_bests]
    labels = tags + ["random\nsearch"]
    bp = ax2.boxplot(data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors + ["#7F8C8D"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=8.5)
    ax2.set_ylabel("Лучшее f(x) за запуск")
    ax2.set_title("Разброс результатов")
    ax2.grid(alpha=0.3, axis="y")

    ax3 = fig.add_subplot(gs[1, :2])
    means = [results[t]["bests"].mean() for t in tags] + [rs_bests.mean()]
    stds = [results[t]["bests"].std() for t in tags] + [rs_bests.std()]
    xpos = np.arange(len(means))
    bars = ax3.bar(xpos, means, yerr=stds, capsize=6,
                   color=colors + ["#7F8C8D"], alpha=0.82, edgecolor="white")
    for bar, m in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=9, color="#12263A")
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(labels, rotation=20, ha="right")
    ax3.set_ylabel("Среднее лучшее f(x)")
    ax3.set_title("Средний результат (чем ниже, тем лучше)")
    ax3.grid(alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    cell_text = []
    for tag in tags:
        b = results[tag]["bests"]
        cell_text.append([tag, f"{b.min():.3g}", f"{b.mean():.3g}", f"{np.median(b):.3g}"])
    cell_text.append(["random", f"{rs_bests.min():.3g}", f"{rs_bests.mean():.3g}",
                      f"{np.median(rs_bests):.3g}"])
    table = ax4.table(cellText=cell_text,
                      colLabels=["конфиг", "min", "mean", "median"],
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2F5D8A")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#F2F6FA")
    ax4.set_title("Сводная статистика (f(x))", pad=18)

    plt.savefig(path, dpi=160, bbox_inches="tight")
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
    save_histories(ga_results, "lab1_out/lab1_histories.npz")
    plot_convergence(ga_results, "lab1_out/lab1_convergence.png")
    plot_comparison(ga_results, "lab1_out/lab1_comparison.png")
    plot_flowchart("lab1_out/lab1_block_schema.png")

    # --- Сравнение со случайным поиском ---
    budget = cfgs[0].pop_size * cfgs[0].max_gens  # равный бюджет оценок
    rs_bests = []
    for r in range(N_RUNS):
        rs = random_search(budget=budget, seed=BASE_SEED + 1000 + r)
        rs_bests.append(rs["best"])
    rs_bests = np.array(rs_bests)
    save_random_csv(rs_bests, "lab1_out/lab1_random.csv")
    plot_dashboard(ga_results, rs_bests, "lab1_out/lab1_dashboard.png")

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