import numpy as np
from typing import Optional, Literal, List, Tuple, Dict

from ..sto_params import STOOrbital
from ..molden import Shell
from .cases import (
    overlap_ss_primitive, overlap_sp_primitive,
    overlap_ps_primitive, overlap_pp_primitive,
)
from .constants import N_s, N_p

Axis = Literal["x", "y", "z"]

def _component(value_3: Tuple[float, float, float], axis: Axis) -> float:
    x, y, z = value_3
    return {"x": x, "y": y, "z": z}[axis]

def overlap_sto_gto(
    sto: STOOrbital,
    gto_shell: Shell,
    A: Tuple[float, float, float],
    B: Tuple[float, float, float],
    axis_sto: Optional[Axis] = None,   # for p-STO: which component (x/y/z)
    axis_gto: Optional[Axis] = None,   # for p-GTO shell: which component (x/y/z)
    quad_n: int = 48,
) -> float:
    """
    Contracted STO–GTO overlap for a *single* GTO shell (s or p) on atom B
    and one STO on atom A. Returns a scalar overlap; for p–p you can call
    multiple times for each (axis_sto, axis_gto) pair, or build a 3x3.
    """
    # geometry
    Rx = A[0] - B[0]
    Ry = A[1] - B[1]
    Rz = A[2] - B[2]
    R2 = Rx*Rx + Ry*Ry + Rz*Rz

    L = gto_shell.L.lower()
    if L not in {"s", "p"}:
        raise NotImplementedError("Only s and p shells are supported in this overlap version.")

    total = 0.0
    for prim in gto_shell.primitives:
        alpha = prim.alpha
        # coefficient for this angular type (Molden 'sp' uses dict; here L is s or p)
        if L not in prim.coeffs:
            # skip if shell label and primitive coeff kind don't match (e.g., 'sp' split)
            coeff = 0.0
        else:
            coeff = prim.coeffs[L]

        if coeff == 0.0:
            continue

        if sto.l == "s" and L == "s":
            total += coeff * overlap_ss_primitive(sto.n, sto.zeta, alpha, R=np.sqrt(R2), quad_n=quad_n)

        elif sto.l == "s" and L == "p":
            if axis_gto is None:
                raise ValueError("axis_gto must be 'x'/'y'/'z' for s–p.")
            Rk = {"x": Rx, "y": Ry, "z": Rz}[axis_gto]
            total += coeff * overlap_sp_primitive(sto.n, sto.zeta, alpha, Rk=Rk, R=np.sqrt(R2), quad_n=quad_n)

        elif sto.l == "p" and L == "s":
            if axis_sto is None:
                raise ValueError("axis_sto must be 'x'/'y'/'z' for p–s.")
            Rk = {"x": Rx, "y": Ry, "z": Rz}[axis_sto]
            total += coeff * overlap_ps_primitive(sto.n, sto.zeta, alpha, Rk=Rk, R=np.sqrt(R2), quad_n=quad_n)

        elif sto.l == "p" and L == "p":
            if axis_sto is None or axis_gto is None:
                raise ValueError("axis_sto and axis_gto must be 'x'/'y'/'z' for p–p.")
            Rk = {"x": Rx, "y": Ry, "z": Rz}[axis_sto]
            Rl = {"x": Rx, "y": Ry, "z": Rz}[axis_gto]
            delta = 1.0 if axis_sto == axis_gto else 0.0
            total += coeff * overlap_pp_primitive(sto.n, sto.zeta, alpha, Rk=Rk, Rl=Rl, delta_kl=delta, R=np.sqrt(R2), quad_n=quad_n)

        else:
            raise NotImplementedError("Unsupported l/L combination.")

    return total