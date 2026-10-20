import numpy as np


def read_numbers(filename, cast=float):
    """ Читает все числа из файла. """
    numbers = []
    with open(filename, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            numbers.extend(cast(x) for x in line.split())
    return numbers


def _as_array(values, cast):
    if cast is float:
        return np.array(values)
    return np.array(values, dtype=object)


def read_system(filename, cast=float):
    """
    n
    матрица n x n
    вектор b
    """
    data = read_numbers(filename, cast=cast)
    n = int(data[0])
    a = _as_array(data[1 : 1 + n * n], cast).reshape(n, n)
    b = _as_array(data[1 + n * n : 1 + n * n + n], cast)
    return a, b


def read_system_eps(filename, cast=float):
    """То же + eps в конце."""
    data = read_numbers(filename, cast=cast)
    n = int(data[0])
    a = _as_array(data[1 : 1 + n * n], cast).reshape(n, n)
    b = _as_array(data[1 + n * n : 1 + n * n + n], cast)
    eps = data[1 + n * n + n]
    return a, b, eps


def read_matrix_eps(filename, cast=float):
    """
    n
    матрица n x n
    eps
    """
    data = read_numbers(filename, cast=cast)
    n = int(data[0])
    a = _as_array(data[1 : 1 + n * n], cast).reshape(n, n)
    eps = data[1 + n * n]
    return a, eps


def read_tridiagonal(filename, cast=float):
    """
    n
    нижняя диагональ (n-1)
    главная (n)
    верхняя (n-1)
    правая часть (n)
    """
    data = read_numbers(filename, cast=cast)
    n = int(data[0])
    i = 1
    lower = _as_array(data[i : i + n - 1], cast)
    i += n - 1
    main = _as_array(data[i : i + n], cast)
    i += n
    upper = _as_array(data[i : i + n - 1], cast)
    i += n - 1
    rhs = _as_array(data[i : i + n], cast)
    return lower, main, upper, rhs


def read_scalar_task(filename):
    """eps, x0_simple, x0_newton, plot_left, plot_right"""
    data = read_numbers(filename)
    return data[0], data[1], data[2], data[3], data[4]


def read_system_task(filename):
    """a, eps, x0_simple (2), x0_newton (2)"""
    data = read_numbers(filename)
    a = data[0]
    eps = data[1]
    x0_simple = np.array(data[2:4])
    x0_newton = np.array(data[4:6])
    return a, eps, x0_simple, x0_newton


def _fmt(x):
    """ Fraction и int — как есть, float — коротко """
    if hasattr(x, "numerator") and hasattr(x, "denominator"):
        return str(x)
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.10g}"
    return str(x)


def print_arr(arr, name=None):
    if name is not None:
        print(f"{name}:")
    arr = np.asarray(arr)
    if arr.ndim == 1:
        print("[\n  " + "\n  ".join(_fmt(x) for x in arr) + "\n]")
    elif arr.ndim == 2:
        for row in arr:
            print("[ " + " ".join(_fmt(x) for x in row) + " ]")
    else:
        raise ValueError("Поддерживаются только 1D и 2D массивы")
    print()
