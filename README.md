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