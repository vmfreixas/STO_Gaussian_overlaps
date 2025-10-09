# src/overlap/build_valence_overlap.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal, Dict

import numpy as np

from src.sto_params import get_sto_params, STOOrbital
from src.overlap.api import overlap_sto_gto

Axis = Literal["x", "y", "z"]

@dataclass(frozen=True)
class STOFn:
    atom_index: int
    element: str
    l: Literal["s", "p"]
    n: int
    zeta: float
    axis: Optional[Axis] = None  # for p only
    label: str = ""              # e.g., "C2 2p_x"

@dataclass(frozen=True)
class GTOFn:
    atom_index: int
    element: str
    L: Literal["s", "p"]
    shell_idx: int               # index into mol.shells_by_atom[atom_index]
    axis: Optional[Axis] = None  # for p only
    label: str = ""              # e.g., "C2 p_x (val)"

def _avg_alpha(shell) -> float:
    return sum(p.alpha for p in shell.primitives) / len(shell.primitives)

def _pick_valence_s_shell(shells_on_atom: List) -> int:
    """Return index (in shells_on_atom) of the more-diffuse s shell (valence)."""
    s_indices = [i for i, sh in enumerate(shells_on_atom) if sh.L.lower() == "s"]
    if not s_indices:
        raise ValueError("No s shells on atom for valence selection.")
    if len(s_indices) == 1:
        return s_indices[0]
    # choose more diffuse (smaller avg alpha)
    best = min(s_indices, key=lambda i: _avg_alpha(shells_on_atom[i]))
    return best

def _pick_p_shell(shells_on_atom: List) -> int:
    """Return index of the (first) p shell (STO-3G has one p block per heavy atom)."""
    for i, sh in enumerate(shells_on_atom):
        if sh.L.lower() == "p":
            return i
    raise ValueError("No p shells on atom for valence selection.")

def _sto_valence_orbitals_for_element(symbol: str) -> Dict[str, STOOrbital]:
    """
    From the registry, pick 'valence' STOs: for each l in {s,p} present,
    choose the orbital with the largest n. Return dict like {'s': STOOrbital, 'p': STOOrbital}.
    """
    params = get_sto_params(symbol)
    result: Dict[str, STOOrbital] = {}
    for l in ("s", "p"):
        cands = [o for o in params.orbitals if o.l == l]
        if not cands:
            continue
        # pick the one with max n (assume it's valence)
        best = max(cands, key=lambda o: o.n)
        result[l] = best
    return result

def build_valence_function_lists(mol) -> Tuple[List[STOFn], List[GTOFn]]:
    """
    Build lists of valence STO functions (expanded to p_x/p_y/p_z) and
    valence GTO functions (valence s + p_x/p_y/p_z) for all atoms in 'mol'.
    """
    sto_funcs: List[STOFn] = []
    gto_funcs: List[GTOFn] = []

    for ai, atom in enumerate(mol.atoms):
        elem = atom.element
        # --- STO valence selection (from registry) ---
        sto_val = _sto_valence_orbitals_for_element(elem)
        # s
        if "s" in sto_val:
            o = sto_val["s"]
            sto_funcs.append(STOFn(
                atom_index=ai, element=elem, l="s", n=o.n, zeta=o.zeta,
                axis=None, label=f"{elem}{ai+1} {o.n}s"
            ))
        # p (expand to components)
        if "p" in sto_val:
            o = sto_val["p"]
            for ax in ("x", "y", "z"):
                sto_funcs.append(STOFn(
                    atom_index=ai, element=elem, l="p", n=o.n, zeta=o.zeta,
                    axis=ax, label=f"{elem}{ai+1} {o.n}p_{ax}"
                ))

        # --- GTO valence selection (from Molden shells) ---
        shells = mol.shells_by_atom[ai]
        # s valence (more diffuse)
        try:
            s_idx = _pick_valence_s_shell(shells)
            gto_funcs.append(GTOFn(
                atom_index=ai, element=elem, L="s", shell_idx=s_idx, axis=None,
                label=f"{elem}{ai+1} s_val"
            ))
        except ValueError:
            pass  # no s shells on H? (H will have one; just being safe)

        # p block (expand to components)
        try:
            p_idx = _pick_p_shell(shells)
            for ax in ("x", "y", "z"):
                gto_funcs.append(GTOFn(
                    atom_index=ai, element=elem, L="p", shell_idx=p_idx, axis=ax,
                    label=f"{elem}{ai+1} p_{ax}"
                ))
        except ValueError:
            pass  # e.g., hydrogen lacks p shells in STO-3G

    return sto_funcs, gto_funcs

def build_valence_overlap_matrix(mol, quad_n: int = 128) -> Tuple[np.ndarray, List[STOFn], List[GTOFn]]:
    """
    Returns:
      S   : (n_STO_val × n_GTO_val) overlap matrix
      rows: list of STOFn (row labels)
      cols: list of GTOFn (col labels)
    Only valence orbitals are included (STO highest-n per l; GTO valence s + p shells).
    """
    sto_funcs, gto_funcs = build_valence_function_lists(mol)

    S = np.zeros((len(sto_funcs), len(gto_funcs)))
    for i, sf in enumerate(sto_funcs):
        A = mol.atoms[sf.atom_index].coords
        # build a minimal STOOrbital to pass (uses n, l, zeta)
        sto = STOOrbital(n=sf.n, l=sf.l, zeta=sf.zeta, label=sf.label)
        for j, gf in enumerate(gto_funcs):
            B = mol.atoms[gf.atom_index].coords
            shell = mol.shells_by_atom[gf.atom_index][gf.shell_idx]
            if sf.l == "s" and gf.L == "s":
                S[i, j] = overlap_sto_gto(sto, shell, A, B, quad_n=quad_n)
            elif sf.l == "s" and gf.L == "p":
                S[i, j] = overlap_sto_gto(sto, shell, A, B, axis_gto=gf.axis, quad_n=quad_n)
            elif sf.l == "p" and gf.L == "s":
                S[i, j] = overlap_sto_gto(sto, shell, A, B, axis_sto=sf.axis, quad_n=quad_n)
            elif sf.l == "p" and gf.L == "p":
                S[i, j] = overlap_sto_gto(sto, shell, A, B, axis_sto=sf.axis, axis_gto=gf.axis, quad_n=quad_n)
            else:
                raise RuntimeError("Unexpected l/L combination.")
    return S, sto_funcs, gto_funcs

def pretty_print_matrix(S: np.ndarray, rows: List[STOFn], cols: List[GTOFn], max_width: int = 12) -> None:
    rlab = [r.label for r in rows]
    clab = [c.label for c in cols]
    # header
    print(" " * max_width, *[f"{c:>{max_width}s}" for c in clab], sep="")
    for i, row in enumerate(S):
        print(f"{rlab[i]:>{max_width}s}", *[f"{v:>{max_width}.6f}" for v in row], sep="")
