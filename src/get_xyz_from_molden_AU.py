from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

BOHR_PER_ANG = 1.889_726_125

# ---------- Data model ----------

@dataclass(frozen=True)
class Primitive:
    alpha: float                     # exponent
    coeffs: Dict[str, float]         # e.g. {"s": cs} or {"p": cp} or {"s": cs, "p": cp}

@dataclass
class Shell:
    L: str                           # "s", "p", "d", "f", or "sp" (case-insensitive)
    scale: float                     # usually 1.0
    primitives: List[Primitive] = field(default_factory=list)

@dataclass(frozen=True)
class Atom:
    element: str
    x: float
    y: float
    z: float

    @property
    def coords(self):
        return (self.x, self.y, self.z)

@dataclass
class Molecule:
    atoms: List[Atom] = field(default_factory=list)
    shells_by_atom: List[List[Shell]] = field(default_factory=list)  # parallel to atoms
    units: str = "AU"  # "AU" or "Angs"

    def add_atom(self, atom: Atom):
        self.atoms.append(atom)
        self.shells_by_atom.append([])

    def add_shell_to_atom(self, atom_index_1based: int, shell: Shell):
        if atom_index_1based < 1 or atom_index_1based > len(self.atoms):
            raise IndexError(f"[GTO] references atom index {atom_index_1based}, "
                             f"but only {len(self.atoms)} atoms were read.")
        self.shells_by_atom[atom_index_1based - 1].append(shell)

# ---------- Parsing utilities ----------

def _to_float(s: str) -> float:
    """Parse numbers that may contain Fortran 'D' exponents."""
    return float(s.replace("D", "E").replace("d", "E"))

def _is_section_header(line: str) -> bool:
    ls = line.strip()
    return ls.startswith("[") and ls.endswith("]") or ls.startswith("[Atoms]")

def _strip_comment(line: str) -> str:
    # Molden doesn’t have a strict comment char, but some writers append notes.
    # Keep simple: return line as-is (left here for future extension).
    return line.rstrip("\n")

# ---------- Core parser ----------

def read_molden(path: str) -> Molecule:
    """
    Read a MOLDEN file into Molecule/Atom/Shell/Primitive structures.
    Parses [Atoms] and [GTO]. Other sections are ignored safely.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [_strip_comment(l) for l in f]

    mol = Molecule()
    i = 0
    n = len(raw_lines)

    # Find sections and parse the ones we need
    while i < n:
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue

        # [Atoms] (AU) or (Angs)
        if line.startswith("[Atoms]"):
            units = "AU"
            if "(Angs)" in line or "(angs)" in line:
                units = "Angs"
            elif "(AU)" in line or "(au)" in line:
                units = "AU"
            mol.units = units
            i = _parse_atoms_block(raw_lines, i + 1, mol)  # returns next index after block
            continue

        # [GTO]
        if line.startswith("[GTO]"):
            i = _parse_gto_block(raw_lines, i + 1, mol)    # returns next index after block
            continue

        # Skip other sections
        if line.startswith("["):
            # Advance until the next section header or EOF
            i += 1
            while i < n and not raw_lines[i].strip().startswith("["):
                i += 1
            continue

        i += 1

    return mol

def _parse_atoms_block(lines: List[str], start: int, mol: Molecule) -> int:
    """
    Parse consecutive atom lines until a new section '[' is reached or blank line precedes [GTO].
    Accepted atom line formats (typical Molden):
      sym  idx  Z   x   y   z
    Some generators may omit idx; we handle both by inspecting field count.
    """
    i = start
    n = len(lines)
    is_ang = (mol.units == "Angs")

    while i < n:
        ls = lines[i].strip()
        if not ls:
            # blank line is allowed; keep scanning unless next is a section
            # lookahead
            j = i + 1
            if j < n and lines[j].strip().startswith("["):
                break
            i += 1
            continue
        if ls.startswith("["):
            break

        parts = ls.split()
        if len(parts) >= 6:
            # sym idx Z x y z
            sym = parts[0]
            # idx = parts[1]; Z = parts[2]  # available if needed
            x = _to_float(parts[3])
            y = _to_float(parts[4])
            z = _to_float(parts[5])
        elif len(parts) == 5:
            # sym Z x y z
            sym = parts[0]
            x = _to_float(parts[2])
            y = _to_float(parts[3])
            z = _to_float(parts[4])
        else:
            raise ValueError(f"Unrecognized [Atoms] line at {i+1}: {lines[i]}")

        if is_ang:
            x *= BOHR_PER_ANG
            y *= BOHR_PER_ANG
            z *= BOHR_PER_ANG

        mol.add_atom(Atom(sym, x, y, z))
        i += 1

    return i  # index of next header or first non-atom line

def _parse_gto_block(lines: List[str], start: int, mol: Molecule) -> int:
    """
    Parse [GTO] blocks:
      <atom_index> [0]
      L  nprim  scale
        alpha  c        (for pure shells)
        ...
      sp nprim scale
        alpha  cs  cp   (for sp shells)
        ...
    Repeats per atom; ends at next section or EOF.
    """
    i = start
    n = len(lines)

    def next_nonempty(k: int) -> int:
        while k < n and not lines[k].strip():
            k += 1
        return k

    while i < n:
        i = next_nonempty(i)
        if i >= n:
            break
        ls = lines[i].strip()
        if ls.startswith("["):
            break
        # Expect an atom index line like "1" or "1 0"
        parts = ls.split()
        if not parts or not parts[0].isdigit():
            # Sometimes there are stray lines; if we encounter a shell header,
            # it means the writer omitted the atom index. That's non-standard; raise.
            raise ValueError(f"Expected atom index line in [GTO] at {i+1}, got: {lines[i]}")
        atom_idx = int(parts[0])
        i += 1

        # Now read shells until the next atom index or section
        while i < n:
            i = next_nonempty(i)
            if i >= n:
                break
            ls = lines[i].strip()
            if not ls:
                i += 1
                continue
            if ls.startswith("["):
                # end of [GTO]
                return i
            # If this line starts with an integer, it's the next atom block
            first = ls.split()[0]
            if first.isdigit():
                break

            # Parse shell header: L nprim scale
            L, nprim, scale, i_next = _parse_shell_header(ls, i)
            i = i_next

            # Parse primitives
            prims: List[Primitive] = []
            for _ in range(nprim):
                if i >= n:
                    raise ValueError(f"Unexpected EOF in [GTO] while reading primitives for atom {atom_idx}")
                fields = lines[i].strip().split()
                if not fields:
                    # allow blank lines between primitives
                    i += 1
                    continue
                # Normalize D exponents
                fields = [f.replace("D", "E").replace("d", "E") for f in fields]
                if L == "sp":
                    if len(fields) < 3:
                        raise ValueError(f"sp primitive expects 3 floats at line {i+1}: {lines[i]}")
                    alpha = float(fields[0])
                    cs = float(fields[1])
                    cp = float(fields[2])
                    prims.append(Primitive(alpha, {"s": cs, "p": cp}))
                else:
                    if len(fields) < 2:
                        raise ValueError(f"{L} primitive expects 2 floats at line {i+1}: {lines[i]}")
                    alpha = float(fields[0])
                    c = float(fields[1])
                    prims.append(Primitive(alpha, {L: c}))
                i += 1

            shell = Shell(L=L, scale=scale, primitives=prims)
            mol.add_shell_to_atom(atom_idx, shell)

        # loop continues for next atom index line

    return i

def _parse_shell_header(line: str, idx: int) -> Tuple[str, int, float, int]:
    """
    Parse a shell header 'L  nprim  scale' on a single line.
    Returns (L_lower, nprim, scale, next_index).
    """
    parts = line.split()
    if len(parts) < 2:
        raise ValueError(f"Malformed shell header at line {idx+1}: {line}")
    L = parts[0].lower()
    if L not in {"s", "p", "d", "f", "sp"}:
        raise ValueError(f"Unknown shell label '{parts[0]}' at line {idx+1}")
    try:
        nprim = int(parts[1])
    except Exception as e:
        raise ValueError(f"Expected integer nprim at line {idx+1}: {line}") from e
    scale = 1.0
    if len(parts) >= 3:
        try:
            scale = _to_float(parts[2])
        except Exception as e:
            raise ValueError(f"Expected float scale at line {idx+1}: {line}") from e
    return L, nprim, scale, idx + 1