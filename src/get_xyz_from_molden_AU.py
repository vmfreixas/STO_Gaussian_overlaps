#   This function reads an xyz file and returns the coordinates in AU as a numpy array.

import numpy as np
import src.data

def get_xyz_from_molden_AU(molden_file):
    angstrom_to_au = 1.8897259886 # Conversion factor from Angstroms to Atomic Units (Bohr)
    with open(molden_file, 'r') as mFile:
        for i, line in enumerate(mFile):
            if i == 0:
                num_atoms = int(line.strip()[0])
    return np.array(coords)