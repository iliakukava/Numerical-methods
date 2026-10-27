"""
2.1. Простая итерация и Ньютон для нелинейного уравнения.
Вариант 12: 3^x - 5*x^2 + 1 = 0
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import read_scalar_task


def f(x):
    return 3**x - 5 * x * x + 1


def df(x):
    return 3**x * math.log(3) - 10 * x


def phi(x):
    # x = sqrt((3^x + 1) / 5) — корень около 0.84
    return math.sqrt((3**x + 1) / 5)


def phi_large(x):
    # x = log(5x^2 - 1) / log(3) — корень около 3.96
    return math.log(5 * x * x - 1) / math.log(3)


def simple_iteration(phi_func, x0, eps, max_iter=100000):
    x = x0
    history = [(0, x, None)]
    for k in range(1, max_iter + 1):
        x_new = phi_func(x)
        err = abs(x_new - x)
        history.append((k, x_new, err))
        x = x_new
        if err < eps:
            return x, k, history
    raise RuntimeError("простая итерация не сошлась")


def newton(x0, eps, max_iter=100000):
    x = x0
    history = [(0, x, None)]
    for k in range(1, max_iter + 1):
        x_new = x - f(x) / df(x)
        err = abs(x_new - x)
        history.append((k, x_new, err))
        x = x_new
        if err < eps:
            return x, k, history
    raise RuntimeError("Ньютон не сошёлся")


def print_history(name, history):
    print(f"\n{name}:")
    print(f"  {'k':>4}  {'x':>14}  {'err':>12}")
    for k, x, err in history:
        err_s = str(err) if err is not None else "-"
        print(f"  {k:4d}  {x:14.10g}  {err_s:>12}")


def save_plot(left, right, roots, path):
    xs = np.linspace(left, right, 800)
    ys = [f(x) for x in xs]

    print("\nтаблица f(x) для выбора начального приближения:")
    for x in np.linspace(left, right, 11):
        print(f"  x={x:6.2f}  f={f(x)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.plot(xs, ys, label="f(x)")
    for r in roots:
        plt.axvline(r, color="red", linestyle="--", alpha=0.7)
        plt.scatter([r], [0], color="red", zorder=5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"график: {path}")


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = str(Path(__file__).resolve().parent / "data" / "task2_1.txt")

    eps, x0_s, x0_n, left, right = read_scalar_task(filename)

    print("2.1. 3^x - 5*x^2 + 1 = 0")
    print(f"eps = {eps}, x0_iter = {x0_s}, x0_newton = {x0_n}")

    x_s, it_s, hist_s = simple_iteration(phi, x0_s, eps)
    x_n, it_n, hist_n = newton(x0_n, eps)

    print(f"\nпростая итерация: x = {x_s}, iters = {it_s}, |f| = {abs(f(x_s))}")
    print(f"Ньютон:           x = {x_n}, iters = {it_n}, |f| = {abs(f(x_n))}")
    print_history("простая итерация", hist_s)
    print_history("Ньютон", hist_n)
    print(f"\nНьютон быстрее: {it_n} против {it_s}")

    x_s2, it_s2, _ = simple_iteration(phi_large, 4.0, eps)
    x_n2, it_n2, _ = newton(4.0, eps)
    print(f"\nвторой корень: iter={x_s2} ({it_s2}), newton={x_n2} ({it_n2})")

    save_plot(left, right, [x_n, x_n2], Path(__file__).resolve().parent / "plots" / "task2_1.png")


if __name__ == "__main__":
    main()
