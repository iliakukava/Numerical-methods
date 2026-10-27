import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import print_arr, read_matrix_eps


def offdiag_norm(a):
    n = a.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                s += a[i, j] ** 2
    return math.sqrt(s)


def max_offdiag(a):
    n = a.shape[0]
    p, q = 0, 1
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i, j]) > best:
                best = abs(a[i, j])
                p, q = i, j
    return p, q


def jacobi(a0, eps, max_iter=100000):
    a = a0.copy().astype(float)
    n = a.shape[0]
    v = np.eye(n)
    history = []

    for it in range(max_iter):
        off = offdiag_norm(a)
        history.append((it, off))
        if off < eps:
            return np.diag(a).copy(), v, it, history

        p, q = max_offdiag(a)
        if abs(a[p, p] - a[q, q]) < 1e-15:
            phi = math.pi / 4
        else:
            phi = 0.5 * math.atan2(2 * a[p, q], a[p, p] - a[q, q])

        c = math.cos(phi)
        s = math.sin(phi)

        app, aqq, apq = a[p, p], a[q, q], a[p, q]
        a[p, p] = c * c * app + 2 * c * s * apq + s * s * aqq
        a[q, q] = s * s * app - 2 * c * s * apq + c * c * aqq
        a[p, q] = a[q, p] = 0.0

        for i in range(n):
            if i != p and i != q:
                aip, aiq = a[i, p], a[i, q]
                a[i, p] = a[p, i] = c * aip + s * aiq
                a[i, q] = a[q, i] = -s * aip + c * aiq

        for i in range(n):
            vip, viq = v[i, p], v[i, q]
            v[i, p] = c * vip + s * viq
            v[i, q] = -s * vip + c * viq

    raise RuntimeError("метод Якоби не сошёлся")


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = str(Path(__file__).resolve().parent / "data" / "task1_4.txt")

    a, eps = read_matrix_eps(filename)
    print(f"eps = {eps}")
    print("A:")
    print_arr(a)

    eigenvalues, v, iters, history = jacobi(a, eps)
    print(f"итераций: {iters}")
    print("собственные значения:")
    print_arr(eigenvalues)
    print("V:")
    print_arr(v)

    lam = np.diag(eigenvalues)
    print(f"||A*V - V*Lambda|| = {np.max(np.abs(a @ v - v @ lam))}")

    print("погрешность от числа итераций:")
    step = max(1, len(history) // 10)
    for i, (it, err) in enumerate(history):
        if i % step == 0 or i == len(history) - 1:
            print(f"  {it:4d}  {err}")


if __name__ == "__main__":
    main()
