import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.io import print_arr, read_tridiagonal


def thomas(lower, main, upper, rhs):
    #Прогонка. Возвращает alpha, beta, x.
    n = len(main)
    alpha = np.zeros(n, dtype=Fraction)
    beta = np.zeros(n, dtype=Fraction)

    alpha[0] = -upper[0] / main[0]
    beta[0] = rhs[0] / main[0]

    for i in range(1, n - 1):
        denom = main[i] + lower[i - 1] * alpha[i - 1]
        alpha[i] = -upper[i] / denom
        beta[i] = (rhs[i] - lower[i - 1] * beta[i - 1]) / denom

    denom = main[n - 1] + lower[n - 2] * alpha[n - 2]
    alpha[n - 1] = Fraction(0)
    beta[n - 1] = (rhs[n - 1] - lower[n - 2] * beta[n - 2]) / denom

    x = np.zeros(n, dtype=Fraction)
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return alpha, beta, x


def residual(lower, main, upper, rhs, x):
    n = len(main)
    r = np.zeros(n, dtype=Fraction)
    r[0] = main[0] * x[0] + upper[0] * x[1] - rhs[0]
    for i in range(1, n - 1):
        r[i] = lower[i - 1] * x[i - 1] + main[i] * x[i] + upper[i] * x[i + 1] - rhs[i]
    r[n - 1] = lower[n - 2] * x[n - 2] + main[n - 1] * x[n - 1] - rhs[n - 1]
    return r


def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = str(Path(__file__).resolve().parent / "data" / "task1_2.txt")

    lower, main, upper, rhs = read_tridiagonal(filename, cast=Fraction)
    alpha, beta, x = thomas(lower, main, upper, rhs)

    print("Нижняя диагональ:")
    print_arr(lower)
    print("Главная диагональ:")
    print_arr(main)
    print("Верхняя диагональ:")
    print_arr(upper)
    print("Правая часть:")
    print_arr(rhs)

    print("Прогоночные коэффициенты:")
    for i in range(len(main)):
        print(f"  alpha[{i + 1}] = {alpha[i]}    beta[{i + 1}] = {beta[i]}")
    print()

    print("x:")
    print_arr(x)

    r = residual(lower, main, upper, rhs, x)
    print("невязка Tx - d:")
    print_arr(r)
    print(f"Tx == d: {(r == 0).all()}")


if __name__ == "__main__":
    main()
