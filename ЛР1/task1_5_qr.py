import cmath
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import print_arr, read_matrix_eps


def qr_decompose(a):
    n = a.shape[0]
    q = np.zeros((n, n))
    r = np.zeros((n, n))
    v = a.copy().astype(float)

    for i in range(n):
        r[i, i] = math.sqrt(v[:, i] @ v[:, i])
        q[:, i] = v[:, i] / r[i, i]
        for j in range(i + 1, n):
            r[i, j] = q[:, i] @ v[:, j]
            v[:, j] = v[:, j] - r[i, j] * q[:, i]
    return q, r


def eigenvalues_from_schur(a, tol=1e-8):
    n = a.shape[0]
    eigs = []
    i = 0
    while i < n:
        if i == n - 1 or abs(a[i + 1, i]) < tol:
            eigs.append(complex(a[i, i]))
            i += 1
        else:
            tr = a[i, i] + a[i + 1, i + 1]
            det = a[i, i] * a[i + 1, i + 1] - a[i, i + 1] * a[i + 1, i]
            d = cmath.sqrt(tr * tr - 4 * det)
            eigs.append((tr + d) / 2)
            eigs.append((tr - d) / 2)
            i += 2
    return eigs


def qr_algorithm(a0, eps, max_iter=10000):
    a = a0.copy().astype(float)
    n = a.shape[0]

    for it in range(1, max_iter + 1):
        q, r = qr_decompose(a)
        a = r @ q

        ok = True
        i = 0
        while i < n - 1:
            if abs(a[i + 1, i]) < eps:
                i += 1
            else:
                if i + 2 < n and abs(a[i + 2, i + 1]) >= eps:
                    ok = False
                    break
                i += 2
        if ok:
            return a, eigenvalues_from_schur(a, tol=eps), it

    return a, eigenvalues_from_schur(a, tol=eps), max_iter


def fmt(z):
    if abs(z.imag) < 1e-10:
        return str(z.real)
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real} {sign} {abs(z.imag)}i"


def run(filename, check_complex=False):
    a, eps = read_matrix_eps(filename)
    print(f"файл: {filename}")
    print(f"eps = {eps}")
    print("A:")
    print_arr(a)

    q, r = qr_decompose(a)
    print("Q:")
    print_arr(q)
    print("R:")
    print_arr(r)
    print("Q * R:")
    print_arr(q @ r)
    print(f"||QR - A|| = {np.max(np.abs(q @ r - a))}")

    a_fin, eigs, iters = qr_algorithm(a, eps)
    print(f"итераций QR-алгоритма: {iters}")
    print("матрица после итераций:")
    print_arr(a_fin)
    print("собственные значения:")
    for i, lam in enumerate(eigs, 1):
        print(f"  lambda[{i}] = {fmt(lam)}")

    if check_complex:
        complex_ones = [z for z in eigs if abs(z.imag) > 1e-8]
        if len(complex_ones) >= 2:
            print("есть комплексно-сопряжённая пара")
    print()


def main():
    base = Path(__file__).resolve().parent / "data"
    if len(sys.argv) > 1:
        run(sys.argv[1])
        return

    print("1.5. QR-разложение и QR-алгоритм\n")
    run(str(base / "task1_5.txt"))
    run(str(base / "task1_5_complex.txt"), check_complex=True)


if __name__ == "__main__":
    main()
