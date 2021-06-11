"""This module ensambles the Fock Matrix using the ERIs, the one center integrals,
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
from mndo_multipole import MultiPole

# Load relevant files (MNDO parameters, ERIs multipoles, etc.)

with open('./SYSTEMS/mndo_params.json', 'r') as mndopar:
    PARAMS = json.load(mndopar)
with open('./SYSTEMS/element.json', 'r') as elems:
    ATOMS = json.load(elems)


# One electron integrals

class OneElectronMatrix():
    """Assembles the one electron integrals for the system of interest"""
    def __init__(self, atoms=None):
        self.atoms = atoms
        self.spinorb = self.__set_spinorb()
        self.h_mndo = self.get_one_center()  #  + self.get_two_center()

    def __set_spinorb(self):
        """Set the spin-orbitals involved in the calculation"""
        sporb = []
        for ats in self.atoms:
            for orbs in ATOMS['elements'][str(ats)][3]:
                sporb.append(str(ats) + "_" + orbs)
        return sporb

    def get_one_center(self):
        """Returns the one-center matrix"""
        spinlen = len(self.spinorb)
        h_one = np.zeros((spinlen, spinlen))
        for miu, labelmu in enumerate(self.spinorb):
            for niu, _ in enumerate(self.spinorb):
                atom_i, orb_i = labelmu.split("_")
                if miu == niu:
                    if "s" in orb_i:
                        h_one[miu, niu] = PARAMS['elements'][atom_i]['uss']
                    else:
                        h_one[miu, niu] = PARAMS['elements'][atom_i]['upp']
                else:
                    h_one[miu, niu] = 0

                klopman_one = 0
                for atom_j in self.atoms:
                    if atom_j != atom_i:
                        klopman_one += MultiPole(atom_i, atom_j,
                                                 bra=("1s", "1s")).eval_eri()

                h_one[miu, niu] += klopman_one

        return h_one

    def get_two_center(self):
        """Returns the two-center matrix"""
        h_two = []
        return np.array(h_two)

# Two electron integrals (ERIs)

# Core-Core repulsion energy

# Fock matrix

# Test for H2


if __name__ == "__main__":
    AB = OneElectronMatrix([1, 1])
    print(AB.h_mndo)
