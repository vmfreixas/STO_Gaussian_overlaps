# src/sto_params.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple

LType = Literal["s", "p", "d", "f"]

@dataclass(frozen=True)
class STOOrbital:
    """
    One Slater orbital specification: principal quantum number n,
    angular momentum l (as 's','p','d','f'), and exponent zeta (ζ).
    Normalization is not stored here; compute from (n, l, ζ) when needed.
    """
    n: int
    l: LType
    zeta: float
    label: str = ""

@dataclass
class STOElementParams:
    """
    STO parameters for one element.
    You can hold multiple STOs (e.g., 1s core, 2s valence, 2p valence).
    """
    symbol: str
    orbitals: List[STOOrbital] = field(default_factory=list)

    def add(self, n: int, l: LType, zeta: float, label: str = "") -> None:
        self.orbitals.append(STOOrbital(n=n, l=l, zeta=zeta, label=label))

# -------- Registry --------

_STO_REGISTRY: Dict[str, STOElementParams] = {}

def register_element(symbol: str, orbitals: List[Tuple[int, LType, float, str | None]]) -> None:
    """
    Bulk register an element with a list of (n, l, zeta, label?) tuples.
    Example:
        register_element("H", [(1, "s", 1.24, "1s")])
    """
    params = STOElementParams(symbol=symbol)
    for n, l, zeta, label in orbitals:
        params.add(n=n, l=l, zeta=zeta, label=label or "")
    _STO_REGISTRY[symbol] = params

def get_sto_params(symbol: str) -> STOElementParams:
    """Retrieve STO parameters for an element symbol (case-insensitive)."""
    params = _STO_REGISTRY.get(symbol) or _STO_REGISTRY.get(symbol.capitalize())
    if params is None:
        raise KeyError(f"No STO parameters registered for element '{symbol}'.")
    return params

def list_registered_elements() -> List[str]:
    return sorted(_STO_REGISTRY.keys())

# from "Michael J. S. Dewar, Eve G. Zoebisch, Eamonn F. Healy, and James J. P. Stewart,
# " AM1: A New General Purpose Quantum Mechanical Molecular Model ", JACS, 107, 3920-3909, (1985) 

register_element("H", [
    (1, "s", 1.88078, "1s"),            # AM1 for s orbital of H in a . u . (Bohr - 1)
])

register_element("C", [
    (1, "s", 0.000000, "1s_core"),      # AM1 has no core orbitals
    (2, "s", 1.808665, "2s_val"),       # AM1 for s orbital of C in a . u . (Bohr - 1)
    (2, "p", 1.685116, "2p_val"),       # AM1 for p orbital of C in a . u . (Bohr - 1)
])
