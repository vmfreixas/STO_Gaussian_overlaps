import numpy as np
from numpy.polynomial.laguerre import laggauss

def laguerre_nodes_weights(n: int):
    """Standard Gauss–Laguerre nodes/weights for ∫_0^∞ e^{-t} f(t) dt."""
    t, w = laggauss(n)  # nodes t>0, weights w>0
    return t, w

def u_of_t(zeta: float, t):
    return (zeta * zeta) / (4.0 * t)

def B_of_t(zeta: float, alpha: float, t):
    # B = u + alpha
    return alpha + (zeta * zeta) / (4.0 * t)

def mu_of_t(zeta: float, alpha: float, t):
    # mu = u*alpha / (u + alpha) = α ζ^2 / (ζ^2 + 4 α t)
    return (alpha * zeta * zeta) / (zeta * zeta + 4.0 * alpha * t)
