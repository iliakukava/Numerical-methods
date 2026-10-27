import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import print_arr, read_system_eps
from task1_1_lu import solve_system


def to_iter_form(a, b):
    #Ax = b -> x = Bx + c
    n = a.shape[0]
    B = np.zeros((n, n))
    c = np.zeros(n)
    for i in range(n):
        c[i] = b[i] / a[i, i]
        for j in range(n):
            if i != j:
                B[i, j] = -a[i, j] / a[i, i]
    return B, c


def simple_iteration(B, c, eps, max_iter=100000):
    x = np.zeros(len(c))
    for k in range(1, max_iter + 1):
        x_new = B @ x + c
        err = np.max(np.abs(x_new - x))
        x = x_new
        if err < eps:
            return x, k
    raise RuntimeError("метод простых итераций не сошёлся")


def seidel(B, c, eps, max_iter=100000):
    x = np.zeros(len(c))
    n = len(c)
    for k in range(1, max_iter + 1):
        x_old = x.copy()
        for i in range(n):
            s = c[i]
            for j in range(n):
                if j != i:
                    s += B[i, j] * x[j]
            x[i] = s
        if np.max(np.abs(x - x_old)) < eps:
            return x, k
    raise RuntimeError("метод Зейделя не сошёлся")


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = str(Path(__file__).resolve().parent / "data" / "task1_3.txt")

    a, b, eps = read_system_eps(filename)
    B, c = to_iter_form(a, b)

    print(f"eps = {eps}")
    print("A:")
    print_arr(a)
    print("b:")
    print_arr(b)
    print("B:")
    print_arr(B)
    print("c:")
    print_arr(c)

    x_j, it_j = simple_iteration(B, c, eps)
    x_s, it_s = seidel(B, c, eps)
    x_ref = np.array([float(v) for v in solve_system(a, b)])

    print(f"простые итерации: {it_j} итераций")
    print_arr(x_j, "x")
    print(f"Зейдель: {it_s} итераций")
    print_arr(x_s, "x")
    print_arr(x_ref, "эталон (LU)")

    print(f"||x_jacobi - x_ref||  = {np.max(np.abs(x_j - x_ref))}")
    print(f"||x_seidel - x_ref||  = {np.max(np.abs(x_s - x_ref))}")
    print(f"||Ax - b|| (Якоби)  = {np.max(np.abs(a @ x_j - b))}")
    print(f"||Ax - b|| (Зейдель) = {np.max(np.abs(a @ x_s - b))}")
    print(f"Зейдель быстрее: {it_s} против {it_j}")


if __name__ == "__main__":
    main()
