import numpy as np
from math import pi
from .laguerre import laguerre_nodes_weights, u_of_t, B_of_t, mu_of_t
from .hermite import hermite_phys
from .constants import N_s, N_p, N_sto_s, N_sto_p, PI

def _kernel_common(t, zeta, alpha, a, b, m):
    """
    Returns the common multiplicative kernel:
    t^(-(a+2)) * (1 + 4 t α / ζ^2)^(-b) * exp(- α R^2 ζ^2 t / (4 t α + ζ^2)) * H_m(√t)
    (the exp part needs R^2 which we multiply outside)
    """
    t = np.asarray(t)
    Hm = hermite_phys(m, np.sqrt(t))
    # factor = t^(-(a+2)) * B^{-b}, where B = α + ζ^2/(4t) = (4α t + ζ^2)/(4t)
    # (1 + 4t α / ζ^2)^(-b) is equivalent to (B * 4t / ζ^2)^(-b), but the compact form is fine:
    B_fac = B_of_t(zeta, alpha, t) ** (-b)
    pow_t = t ** (-(a + 2.0))
    pref = (zeta * zeta / 4.0) ** (a + 1.0)
    return pref * pow_t * B_fac * Hm  # missing exp(-mu R^2), multiply later with R^2

def overlap_ss_primitive(n, zeta, alpha, R, quad_n=48):
    # a, b, m, C
    a = -0.5 - 0.5 * n
    b = 1.5
    m = n
    C = N_sto_s(n, zeta) * N_s(alpha) * PI * (2.0 ** (-n))

    t, w = laguerre_nodes_weights(quad_n)
    ker = _kernel_common(t, zeta, alpha, a, b, m)
    expo = np.exp(-mu_of_t(zeta, alpha, t) * (R * R))
    G = ker * expo
    return C * np.dot(w, G)

def overlap_sp_primitive(n, zeta, alpha, Rk, R, quad_n=48):
    # a, b, m, C (note Rk in C)
    a = +0.5 - 0.5 * n
    b = 2.5
    m = n
    C = N_sto_s(n, zeta) * N_p(alpha) * PI * (2.0 ** (-n)) * Rk

    t, w = laguerre_nodes_weights(quad_n)
    ker = _kernel_common(t, zeta, alpha, a, b, m)
    expo = np.exp(-mu_of_t(zeta, alpha, t) * (R * R))
    G = ker * expo
    return C * np.dot(w, G)

def overlap_ps_primitive(n, zeta, alpha, Rk, R, quad_n=48):
    # a, b, m, C (note -α Rk in C)
    a = -0.5 * n
    b = 2.5
    m = n - 1
    C = - N_sto_p(n, zeta) * N_s(alpha) * PI * (2.0 ** (-(n - 1))) * alpha * Rk

    t, w = laguerre_nodes_weights(quad_n)
    ker = _kernel_common(t, zeta, alpha, a, b, m)
    expo = np.exp(-mu_of_t(zeta, alpha, t) * (R * R))
    G = ker * expo
    return C * np.dot(w, G)

def overlap_pp_primitive(n, zeta, alpha, Rk, Rl, delta_kl, R, quad_n=48):
    # a, b, m, C
    a = -0.5 * n
    b = 1.5
    m = n - 1
    C = N_sto_p(n, zeta) * N_p(alpha) * PI * (2.0 ** (-(n - 1)))

    t, w = laguerre_nodes_weights(quad_n)
    ker = _kernel_common(t, zeta, alpha, a, b, m)
    expo = np.exp(-mu_of_t(zeta, alpha, t) * (R * R))

    # Angular term in t:
    # A_kl(t) = -(4 α ζ^2 t)/(4 α t + ζ^2)^2 * Rk Rl  + (2 t)/(4 α t + ζ^2) δ_kl
    denom = (4.0 * alpha * t + zeta * zeta)
    shift = - (4.0 * alpha * zeta * zeta * t) / (denom * denom) * (Rk * Rl)
    delta_term = (2.0 * t / denom) * delta_kl
    Akl = shift + delta_term

    G = ker * expo * Akl
    return C * np.dot(w, G)
