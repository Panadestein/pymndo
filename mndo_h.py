"""This module assembles the Symmetrically Orthogonalized One-Electron
Matrix for the MNDO method:

    Dewar, M. J. S.; Thiel, W. Ground States of Molecules. 38. The MNDO
    Method. Approximations and Parameters, J. Am. Chem. Soc. 1977,
    99, 4899–4907.

    Husch, T., Vaucher, A. C., & Reiher, M. (2018). Semiempirical molecular
    orbital models based on the neglect of diatomic differential overlap
    approximation. International journal of quantum chemistry, 118(24), e25799.

"""
import json
import numpy as np

with open('./SYSTEMS/mndo_params.json', 'r') as mndopar:
    PARAMS = json.load(mndopar)
with open('./SYSTEMS/element.json', 'r') as elems:
    ATOMS = json.load(elems)

class OneElectronMatrix():
    """Assembles the one electron integrals for the system of interest"""
    def __init__(self, atoms=None):
        self.atoms = atoms
        self.h_mndo = self.get_one_center() + self.get_two_center()
        self.spinorb = self.__set_spinorb()
    def __set_spinorb(self):
        """Set the spin-orbitals involved in the calculation"""
        sporb = []
        for ats in self.atoms:
            for orbs in ATOMS['element'][str(ats)][3]:
                sporb.append(str(ats) + "_" + orbs)
        return sporb
    def get_one_center(self):
        """Returns the one-center matrix"""
        h_one = np.zeros((self.spinorb, self.spinorb))
        for miu, labelmu in enumerate(self.spinorb):
            for niu, _ in enumerate(self.spinorb):
                if miu == niu:
                    atom_i, orb_i = labelmu.split("_")
                    if "s" in orb_i:
                        h_one[miu, niu] = PARAMS['elements'][atom_i]['uss']
                    else:
                        h_one[miu, niu] = PARAMS['elements'][atom_i]['upp']
                else:
                    h_one[miu, niu] = 0
    def get_two_center(self):
        """Returns the two-center matrix"""
        h_two = []
        return np.array(h_two)


def klopman(miu, niu, z_atom):
    """Calculates an ERI using the Klopman approximation"""
