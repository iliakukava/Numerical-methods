"""
2.2. Простая итерация и Ньютон для системы нелинейных уравнений.
Вариант 12, a = 3:
  x1 - cos(x2) = a
  x2 - sin(x1) = a
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import print_arr, read_system_task


def F(x, a):
    return np.array([
        x[0] - math.cos(x[1]) - a,
        x[1] - math.sin(x[0]) - a,
    ])


def jacobian(x):
    return np.array([
        [1.0, math.sin(x[1])],
        [-math.cos(x[0]), 1.0],
    ])


def phi(x, a):
    return np.array([a + math.cos(x[1]), a + math.sin(x[0])])


def solve_2x2(A, b):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return np.array([
        (b[0] * A[1, 1] - b[1] * A[0, 1]) / det,
        (A[0, 0] * b[1] - A[1, 0] * b[0]) / det,
    ])


def simple_iteration(a, x0, eps, max_iter=100000):
    x = x0.copy().astype(float)
    history = [(0, x.copy(), None)]
    for k in range(1, max_iter + 1):
        x_new = phi(x, a)
        err = np.max(np.abs(x_new - x))
        history.append((k, x_new.copy(), err))
        x = x_new
        if err < eps:
            return x, k, history
    raise RuntimeError("простая итерация не сошлась")


def newton(a, x0, eps, max_iter=100000):
    x = x0.copy().astype(float)
    history = [(0, x.copy(), None)]
    for k in range(1, max_iter + 1):
        x_new = x + solve_2x2(jacobian(x), -F(x, a))
        err = np.max(np.abs(x_new - x))
        history.append((k, x_new.copy(), err))
        x = x_new
        if err < eps:
            return x, k, history
    raise RuntimeError("Ньютон не сошёлся")


def print_history(name, history):
    print(f"\n{name}:")
    print(f"  {'k':>4}  {'x1':>14}  {'x2':>14}  {'err':>12}")
    for k, x, err in history:
        err_s = str(err) if err is not None else "-"
        print(f"  {k:4d}  {x[0]:14.10g}  {x[1]:14.10g}  {err_s:>12}")


def save_plot(a, sol, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(a - 2, a + 2, 400)
    ys = np.linspace(a - 2, a + 2, 400)
    X, Y = np.meshgrid(xs, ys)

    plt.figure(figsize=(7, 6))
    plt.contour(X, Y, X - np.cos(Y) - a, levels=[0], colors="tab:blue")
    plt.contour(X, Y, Y - np.sin(X) - a, levels=[0], colors="tab:orange")
    plt.scatter([sol[0]], [sol[1]], color="red", zorder=5, label="решение")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"график: {path}")


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = str(Path(__file__).resolve().parent / "data" / "task2_2.txt")

    a, eps, x0_s, x0_n = read_system_task(filename)

    print("2.2. система: x1 - cos(x2) = a, x2 - sin(x1) = a")
    print(f"a = {a}, eps = {eps}")
    print_arr(x0_s, "x0 (итерации)")
    print_arr(x0_n, "x0 (Ньютон)")

    x_s, it_s, hist_s = simple_iteration(a, x0_s, eps)
    x_n, it_n, hist_n = newton(a, x0_n, eps)

    print(f"простая итерация: ({x_s[0]}, {x_s[1]}), iters = {it_s}")
    print(f"  ||F|| = {np.max(np.abs(F(x_s, a)))}")
    print(f"Ньютон:           ({x_n[0]}, {x_n[1]}), iters = {it_n}")
    print(f"  ||F|| = {np.max(np.abs(F(x_n, a)))}")

    print_history("простая итерация", hist_s)
    print_history("Ньютон", hist_n)
    print(f"\nНьютон быстрее: {it_n} против {it_s}")

    save_plot(a, x_n, Path(__file__).resolve().parent / "plots" / "task2_2.png")

    print(f"\nx1 - cos(x2) = {x_n[0] - math.cos(x_n[1])}  (a={a})")
    print(f"x2 - sin(x1) = {x_n[1] - math.sin(x_n[0])}  (a={a})")


if __name__ == "__main__":
    main()
