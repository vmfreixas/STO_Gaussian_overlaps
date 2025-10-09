import math

PI = math.pi

def N_s(alpha: float) -> float:
    # s-type primitive GTO normalization
    return (2.0 * alpha / PI) ** 0.75

def N_p(alpha: float) -> float:
    # p-type primitive GTO normalization (Cartesian)
    return 2.0 ** 0.5 * (2.0 * alpha) ** 1.25 * PI ** (-0.75)

def N_sto_s(n: int, zeta: float) -> float:
    # STO s (real Y_00): includes 1/sqrt(4π)
    # ((2ζ)^(2n+1)/(2n)!)^1/2 * 1/sqrt(4π)
    from math import factorial
    return ((2.0 * zeta) ** (2 * n + 1) / factorial(2 * n)) ** 0.5 / (4.0 * PI) ** 0.5

def N_sto_p(n: int, zeta: float) -> float:
    # STO p (Cartesian real): sqrt(3/4π) factor
    from math import factorial
    return ((2.0 * zeta) ** (2 * n + 1) / factorial(2 * n)) ** 0.5 * (3.0 / (4.0 * PI)) ** 0.5
