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
from common.io import read_numbers


def f(x):
    return 3**x - 5 * x * x + 1


def df(x):
    return 3**x * math.log(3) - 10 * x


def d2f(x):
    return 3**x * math.log(3) ** 2 - 10


def phi(x):
    # x = sqrt((3^x + 1) / 5) — меньший корень
    return math.sqrt((3**x + 1) / 5)


def dphi(x):
    return 3**x * math.log(3) / (10 * phi(x))


def phi_large(x):
    # x = log(5x^2 - 1) / log(3) — больший корень
    return math.log(5 * x * x - 1) / math.log(3)


def dphi_large(x):
    return 10 * x / (math.log(3) * (5 * x * x - 1))


def samples(a, b, n=1000):
    return [a + i * (b - a) / (n - 1) for i in range(n)]


def check_simple(a, b, phi_func, dphi_func):
    """Теорема: phi([a, b]) лежит в [a, b] и |phi'| <= q < 1."""
    xs = samples(a, b)
    values = [phi_func(x) for x in xs]
    q = max(abs(dphi_func(x)) for x in xs)
    inside = min(values) >= a and max(values) <= b
    print(f" phi([a, b]) в [{min(values):.6g}, {max(values):.6g}], внутри отрезка: {inside}")
    print(f" q = max |phi'| = {q}")
    if not inside or q >= 1:
        raise RuntimeError("достаточные условия сходимости простой итерации не выполнены")
    return q


def check_newton(a, b):
    """Теорема: смена знака f, постоянный знак f' и f'', условие Фурье."""
    xs = samples(a, b)
    d1 = [df(x) for x in xs]
    d2 = [d2f(x) for x in xs]
    sign_change = f(a) * f(b) < 0
    d1_const = d1[0] * d1[-1] > 0 and all(v * d1[0] > 0 for v in d1)
    d2_const = d2[0] * d2[-1] > 0 and all(v * d2[0] > 0 for v in d2)
    print(f" f(a)*f(b) < 0: {sign_change}")
    print(f" f' не меняет знак: {d1_const}")
    print(f" f'' не меняет знак: {d2_const}")
    if not (sign_change and d1_const and d2_const):
        raise RuntimeError("достаточные условия сходимости Ньютона не выполнены")

    if f(a) * d2f(a) > 0:
        return a
    if f(b) * d2f(b) > 0:
        return b
    raise RuntimeError("условие Фурье не выполнено ни на одном конце отрезка")


def simple_iteration(phi_func, x0, eps, q, max_iter=100000):
    # оценка погрешности: q/(1-q) * |x_{k+1} - x_k|
    factor = q / (1 - q)
    x = x0
    history = [(0, x, None)]
    for k in range(1, max_iter + 1):
        x_new = phi_func(x)
        err = factor * abs(x_new - x)
        history.append((k, x_new, err))
        x = x_new
        if err <= eps:
            return x, k, history
    raise RuntimeError("простая итерация не сошлась")


def newton(x0, eps, max_iter=100000):
    x = x0
    history = [(0, x, None)]
    for k in range(1, max_iter + 1):
        deriv = df(x)
        if abs(deriv) < 1e-15:
            raise ValueError("f'(x) = 0, касательная горизонтальна")
        x_new = x - f(x) / deriv
        err = abs(x_new - x)
        history.append((k, x_new, err))
        x = x_new
        if err < eps:
            return x, k, history
    raise RuntimeError("Ньютон не сошёлся")


def print_history(name, history):
    print(f"\n{name}:")
    print(f" {'k':>4}  {'x':>14}  {'err':>12}")
    for k, x, err in history:
        err_s = str(err) if err is not None else "-"
        print(f" {k:4d}  {x:14.10g}  {err_s:>12}")


def solve_on_interval(title, a, b, phi_func, dphi_func, eps):
    print(f"\n{title}, отрезок [{a}, {b}]")
    print("простая итерация:")
    q = check_simple(a, b, phi_func, dphi_func)
    x0_s = (a + b) / 2
    print(f"  x0 = (a+b)/2 = {x0_s}")

    print("Ньютон:")
    x0_n = check_newton(a, b)
    print(f"  x0 = {x0_n} (конец, где f*f'' > 0)")

    x_s, it_s, hist_s = simple_iteration(phi_func, x0_s, eps, q)
    x_n, it_n, hist_n = newton(x0_n, eps)
    print(f"простая итерация: x = {x_s}, iters = {it_s}, |f| = {abs(f(x_s))}")
    print(f"Ньютон:           x = {x_n}, iters = {it_n}, |f| = {abs(f(x_n))}")
    print_history("простая итерация", hist_s)
    print_history("Ньютон", hist_n)
    print(f"Ньютон быстрее: {it_n} против {it_s}")
    return x_n


def save_plot(left, right, roots, path):
    xs = np.linspace(left, right, 800)
    ys = [f(x) for x in xs]

    print("\nтаблица f(x) для выбора отрезка локализации:")
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

    data = read_numbers(filename)
    eps = data[0]
    a1, b1 = data[1], data[2]
    a2, b2 = data[3], data[4]
    left, right = data[5], data[6]

    print("2.1. 3^x - 5*x^2 + 1 = 0")
    print(f"eps = {eps}")

    x1 = solve_on_interval("меньший корень", a1, b1, phi, dphi, eps)
    x2 = solve_on_interval("больший корень", a2, b2, phi_large, dphi_large, eps)

    save_plot(left, right, [x1, x2], Path(__file__).resolve().parent / "plots" / "task2_1.png")


if __name__ == "__main__":
    main()
