# STO_Gaussian_overlaps
This package calculates the overlaps between Slater Type Orbitals and Gaussian Type Orbitals, in particular STO-3G.

It reads a molden file to get atomic coordinates and Gaussian basis coefficients.
STO coefficient correspond to AM1. 
In the current implementation only s and p valence orbitals are included. 

The matrix contains STO in the rows and GTO in the columns.

A working example can be found in notebooks/overlap_matrix.ipynb
- Here the overlap matrix is calculated from reading the molden file
- The matrix is plotted 
- The matrix is saved to a txt file

# The package also calculates several densities in the AO basis from NEXMD:
- It reads
    - The MO matrix from the "vhf.out" file
    - Transition density matrices from the GS to the excited states from the "transition-densities.put" file
- It calculates in the AO
    - The GS density matrix
    - The excited states density
    - The transition density matrices between excited states
- The calulcated densities are demosntrated in the notebook "notebooks/densities.ipnb", which also plots and save them to a file 