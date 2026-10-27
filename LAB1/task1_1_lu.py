import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import print_arr, read_system


def lu_decompose(a: np.ndarray):
    #PA = LU. Возвращает L, U, P и число перестановок строк.
    n = a.shape[0]
    u = a.copy().astype(Fraction)
    l = np.identity(n, dtype=Fraction)
    p = np.identity(n, dtype=Fraction)
    swaps = 0

    for k in range(n - 1):
        pivot = k + int(np.abs(u[k:, k]).argmax())
        if u[pivot, k] == Fraction(0):
            raise ValueError("Матрица вырождена")

        if pivot != k:
            swaps += 1
            u[[k, pivot]] = u[[pivot, k]]
            p[[k, pivot]] = p[[pivot, k]]
            if k > 0:
                l[[k, pivot], :k] = l[[pivot, k], :k]

        for i in range(k + 1, n):
            l[i, k] = u[i, k] / u[k, k]
            for j in range(k, n):
                u[i, j] = u[i, j] - l[i, k] * u[k, j]

    if u[n - 1, n - 1] == Fraction(0):
        raise ValueError("Матрица вырождена")
    
    return l, u, p, swaps


def forward(l: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = l.shape[0]
    y = np.zeros(n, dtype=Fraction)
    for i in range(n):
        y[i] = b[i] - l[i, :i] @ y[:i]
    return y


def backward(u: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = u.shape[0]
    x = np.zeros(n, dtype=Fraction)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - u[i, i + 1 :] @ x[i + 1 :]) / u[i, i]
    return x


def solve_lu(l, u, p, b):
    return backward(u, forward(l, p @ b))


def inverse_from_lu(l, u, p):
    n = l.shape[0]
    inv = np.zeros((n, n), dtype=Fraction)
    for j in range(n):
        e = np.zeros(n, dtype=Fraction)
        e[j] = Fraction(1)
        inv[:, j] = solve_lu(l, u, p, e)
    return inv


def det_from_lu(u, swaps):
    det = Fraction(1)
    for i in range(u.shape[0]):
        det *= u[i, i]
    return det if swaps % 2 == 0 else -det


def solve_system(a, b):
    a_f = np.array([[Fraction(x) for x in row] for row in np.asarray(a)], dtype=object)
    b_f = np.array([Fraction(x) for x in np.asarray(b)], dtype=object)
    l, u, p, _ = lu_decompose(a_f)
    return solve_lu(l, u, p, b_f)


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = str(Path(__file__).resolve().parent / "data" / "task1_1.txt")

    a, b = read_system(filename, cast=Fraction)
    l, u, p, swaps = lu_decompose(a)
    x = solve_lu(l, u, p, b)
    a_inv = inverse_from_lu(l, u, p)
    det = det_from_lu(u, swaps)

    print("L:")
    print_arr(l)
    print("U:")
    print_arr(u)

    lu = l @ u
    print("L * U:")
    print_arr(lu)

    print("x:")
    print_arr(x)

    print("A^(-1):")
    print_arr(a_inv)
    print(f"det(A) = {det}")

    print("A * A^(-1):")
    aa_inv = a @ a_inv
    print_arr(aa_inv)

    pa = p @ a
    print(f"L*U == P*A: {(lu == pa).all()}")
    print(f"A*A^(-1) == E: {(aa_inv == np.identity(a.shape[0], dtype=Fraction)).all()}")
    print(f"Ax == b: {(a @ x == b).all()}")


if __name__ == "__main__":
    main()
