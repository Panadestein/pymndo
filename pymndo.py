"""
PyMNDO: A simple pythonic MNDO method implementation. Currently only atoms
from first and second period are supported, but the code can be easily extended
upon availability of semiempirical parameters.
"""
from molecule import Molecule
from mndo_h import OneElectronMatrix

# Define a test system (Water molecule)

H2O = Molecule("H2)", [(8, (0.00000000, 0.00000000, 0.04851804)),
                       (1, (0.75300223, 0.00000000, -0.51923377)),
                       (1, (-0.75300223, 0.00000000, -0.51923377))],
               charge=0, multiplicity=1)

# One electron integrals

HMAT = OneElectronMatrix(H2O.atoms)
print(HMAT.spinorb)
