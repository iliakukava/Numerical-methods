import cmath
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import print_arr, read_matrix_eps


def qr_decompose(a):
    n = a.shape[0]
    r = a.copy().astype(float)
    q = np.eye(n)

    for k in range(n - 1):
        x = r[k:, k].copy()
        norm = math.sqrt(x @ x)
        if norm == 0:
            continue

        sign = 1.0 if x[0] >= 0 else -1.0
        e1 = np.zeros(len(x))
        e1[0] = 1.0
        v = x + sign * norm * e1

        h_small = np.eye(len(x)) - 2 * np.outer(v, v) / (v @ v)
        h = np.eye(n)
        h[k:, k:] = h_small

        r = h @ r
        q = q @ h

    return q, r


def block_eigenvalues(a, eps):
    n = a.shape[0]
    eigs = []
    i = 0
    while i < n:
        if i == n - 1 or abs(a[i + 1, i]) < eps:
            eigs.append(complex(a[i, i]))
            i += 1
        elif i + 1 == n - 1 or abs(a[i + 2, i + 1]) < eps:
            tr = a[i, i] + a[i + 1, i + 1]
            det = a[i, i] * a[i + 1, i + 1] - a[i, i + 1] * a[i + 1, i]
            d = cmath.sqrt(tr * tr - 4 * det)
            eigs.append((tr + d) / 2)
            eigs.append((tr - d) / 2)
            i += 2
        else:
            return None
    return eigs


def eigenvalues_close(prev, curr, eps):
    prev = sorted(prev, key=lambda z: (z.real, z.imag))
    curr = sorted(curr, key=lambda z: (z.real, z.imag))
    for old, new in zip(prev, curr):
        if abs(new - old) > eps:
            return False
    return True


def qr_algorithm(a0, eps, max_iter=10000):
    #Остановка, когда СЗ блоков перестали меняться: |λ^k - λ^{k-1}| ≤ ε.
    a = a0.copy().astype(float)
    prev = block_eigenvalues(a, eps)

    for it in range(1, max_iter + 1):
        q, r = qr_decompose(a)
        a = r @ q
        eigs = block_eigenvalues(a, eps)
        if eigs is not None and prev is not None and eigenvalues_close(prev, eigs, eps):
            return a, eigs, it
        if eigs is not None:
            prev = eigs

    eigs = block_eigenvalues(a, eps)
    return a, eigs if eigs is not None else [], max_iter


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
