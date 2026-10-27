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
from common.io import print_arr, read_numbers


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


def dphi(x):
    """Матрица φ'(x)."""
    return np.array([
        [0.0, -math.sin(x[1])],
        [math.cos(x[0]), 0.0],
    ])


def row_norm(matrix):
    return max(np.sum(np.abs(row)) for row in matrix)


def in_box(x, box):
    x1_left, x1_right, x2_left, x2_right = box
    return x1_left <= x[0] <= x1_right and x2_left <= x[1] <= x2_right


def check_simple(a, x0, r):
    """Теорема: φ(G) лежит в G и q < 1."""
    box = (x0[0] - r[0], x0[0] + r[0], x0[1] - r[1], x0[1] + r[1])
    xs = np.linspace(box[0], box[1], 40)
    ys = np.linspace(box[2], box[3], 40)
    q = 0.0
    inside = True
    for x1 in xs:
        for x2 in ys:
            point = np.array([x1, x2])
            q = max(q, row_norm(dphi(point)))
            if not in_box(phi(point, a), box):
                inside = False
    print(f"  G = [{box[0]:.4g}, {box[1]:.4g}] x [{box[2]:.4g}, {box[3]:.4g}]")
    print(f"  phi(G) внутри G: {inside}")
    print(f"  q = {q}")
    if not inside or q >= 1:
        raise RuntimeError("достаточные условия сходимости простой итерации не выполнены")
    return q, box


def solve_2x2(A, b):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-15:
        raise ValueError("Матрица Якоби вырождена (det J = 0)")
    return np.array([
        (b[0] * A[1, 1] - b[1] * A[0, 1]) / det,
        (A[0, 0] * b[1] - A[1, 0] * b[0]) / det,
    ])


def simple_iteration(a, x0, eps, q, box, max_iter=100000):
    # оценка: q/(1-q) * ||x_{k+1} - x_k||, ||.|| = max |x_i|
    factor = q / (1 - q)
    x = x0.copy().astype(float)
    history = [(0, x.copy(), None)]
    for k in range(1, max_iter + 1):
        x_new = phi(x, a)
        if not in_box(x_new, box):
            raise RuntimeError("итерация вышла за пределы области G")
        err = factor * np.max(np.abs(x_new - x))
        history.append((k, x_new.copy(), err))
        x = x_new
        if err <= eps:
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


def save_plot(a, sol, x0, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(a - 2, a + 2, 400)
    ys = np.linspace(a - 2, a + 2, 400)
    X, Y = np.meshgrid(xs, ys)

    plt.figure(figsize=(7, 6))
    plt.contour(X, Y, X - np.cos(Y) - a, levels=[0], colors="tab:blue")
    plt.contour(X, Y, Y - np.sin(X) - a, levels=[0], colors="tab:orange")
    plt.plot([], [], color="tab:blue", label="x1 - cos(x2) = a")
    plt.plot([], [], color="tab:orange", label="x2 - sin(x1) = a")
    plt.scatter([x0[0]], [x0[1]], color="black", zorder=5, label="x0")
    plt.scatter([sol[0]], [sol[1]], color="red", zorder=6, label="решение")
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

    data = read_numbers(filename)
    a = data[0]
    eps = data[1]
    x0_s = np.array(data[2:4])
    x0_n = np.array(data[4:6])
    r = np.array(data[6:8])

    print("2.2. система: x1 - cos(x2) = a, x2 - sin(x1) = a")
    print(f"a = {a}, eps = {eps}")
    print_arr(x0_s, "x0 (итерации)")
    print_arr(x0_n, "x0 (Ньютон)")

    print("простая итерация:")
    q, box = check_simple(a, x0_s, r)
    x_s, it_s, hist_s = simple_iteration(a, x0_s, eps, q, box)
    x_n, it_n, hist_n = newton(a, x0_n, eps)

    print(f"простая итерация: ({x_s[0]}, {x_s[1]}), iters = {it_s}")
    print(f"  ||F|| = {np.max(np.abs(F(x_s, a)))}")
    print(f"Ньютон:           ({x_n[0]}, {x_n[1]}), iters = {it_n}")
    print(f"  ||F|| = {np.max(np.abs(F(x_n, a)))}")

    print_history("простая итерация", hist_s)
    print_history("Ньютон", hist_n)
    print(f"\nНьютон быстрее: {it_n} против {it_s}")

    save_plot(a, x_n, x0_s, Path(__file__).resolve().parent / "plots" / "task2_2.png")

    print(f"\nx1 - cos(x2) = {x_n[0] - math.cos(x_n[1])}  (a={a})")
    print(f"x2 - sin(x1) = {x_n[1] - math.sin(x_n[0])}  (a={a})")


if __name__ == "__main__":
    main()
