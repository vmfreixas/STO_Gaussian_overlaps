import numpy as np

def hermite_phys(n: int, x: np.ndarray) -> np.ndarray:
    """
    Physicists' Hermite polynomials H_n(x): H0=1, H1=2x, H_{n+1}=2x H_n - 2n H_{n-1}
    Works vectorized on x (numpy array).
    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2.0 * x
    Hm2 = np.ones_like(x)        # H0
    Hm1 = 2.0 * x                # H1
    for k in range(1, n):
        H = 2.0 * x * Hm1 - 2.0 * k * Hm2
        Hm2, Hm1 = Hm1, H
    return Hm1
