"""This module ensembles the Fock Matrix using the ERIs, the one center integrals,
and the core-core repulsion energy for the MNDO method:

    Dewar, M. J. S.; Thiel, W. Ground States of Molecules. 38. The MNDO
    Method. Approximations and Parameters, J. Am. Chem. Soc. 1977,
    99, 4899–4907.

    Husch, T., Vaucher, A. C., & Reiher, M. (2018). Semiempirical molecular
    orbital models based on the neglect of diatomic differential overlap
    approximation. International journal of quantum chemistry, 118(24), e25799.

"""
import json
import numpy as np
from scipy.linalg import block_diag
from molecule import Molecule
from mndo_multipole import MultiPole
from cgto_overlap import s_matrix

np.set_printoptions(linewidth=np.nan)

# Load relevant files (MNDO parameters, ERIs multipoles, etc.)

with open('./SYSTEMS/mndo_params.json', 'r') as mndopar:
    PARAMS = json.load(mndopar)
with open('./SYSTEMS/element.json', 'r') as elems:
    ATOMS = json.load(elems)


# One electron integrals

class OneElectronMatrix():
    """Assembles the one electron integrals for the system of interest"""
    def __init__(self, molecule=None):
        self.molecule = molecule
        self.atoms = molecule.atoms
        self.h_mndo = self.get_one_center() + self.get_two_center()

    def get_one_center(self):
        """Returns the one-center matrix"""
        hmats = []
        for atom_i in self.atoms:
            orbs = ATOMS['elements'][str(atom_i)][3]
            h_one = np.zeros((len(orbs), len(orbs)))
            for miu, labelmu in enumerate(orbs):
                for niu, labelnu in enumerate(orbs):
                    orb_i = labelmu.split("_")[0]
                    if miu == niu:
                        if "s" in orb_i:
                            h_one[miu, niu] = PARAMS['elements'][str(atom_i)]['uss']
                        else:
                            h_one[miu, niu] = PARAMS['elements'][str(atom_i)]['upp']
                        klopman_one = 0
                        for atom_j in self.atoms:
                            q_j = ATOMS['elements'][str(atom_i)][2]
                            if atom_j != atom_i:
                                klopman_one -= q_j * MultiPole(atom_i, atom_j,
                                                         bra=(labelmu, labelnu),
                                                         ket=("1s", "1s")).eval_eri()
                        h_one[miu, niu] += klopman_one
                    else:
                        h_one[miu, niu] = 0

            hmats.append(h_one)
        return block_diag(*hmats)

    def get_two_center(self):
        """Returns the two-center matrix"""
        smat = s_matrix(self.molecule)
        row = 0
        for atom_i in self.atoms:
            orbs_i = ATOMS['elements'][str(atom_i)][3]
            for labelmu in orbs_i:
                col = 0
                for atom_j in self.atoms:
                    orbs_j = ATOMS['elements'][str(atom_j)][3]
                    for labelnu in orbs_j:
                        if atom_i != atom_j:
                            if "s" in labelmu:
                                betamu = PARAMS['elements'][str(atom_i)]['uss']
                            else:
                                betamu = PARAMS['elements'][str(atom_i)]['upp']
                            if "s" in labelnu:
                                betanu = PARAMS['elements'][str(atom_j)]['uss']
                            else:
                                betanu = PARAMS['elements'][str(atom_j)]['upp']
                            smat[row, col] *= 0.5 * (betamu + betanu)
                        else:
                            smat[row, col] = 0
                        col += 1
                row += 1
        return smat

# Two electron integrals (ERIs)

# Core-Core repulsion energy

# Fock matrix

# Test for H2


if __name__ == "__main__":
    # Test with the H2O molecule
    H2O_MOL = Molecule('H2O', [(8, (0.00000000, 0.00000000, 0.04851804)),
                               (1, (0.75300223, 0.00000000, -0.51923377)),
                               (1, (-0.75300223, 0.00000000, -0.51923377))],
                       charge=1, multiplicity=2)
    AB = OneElectronMatrix(H2O_MOL)
    print(AB.get_one_center())
    print(AB.get_two_center())
    print(AB.h_mndo)
