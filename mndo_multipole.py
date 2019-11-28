"""This module contains the classes needed to stablish the multipolar expansion
used to approximate the ERIs

    Dewar, M. J. S.; Thiel, W. Ground States of Molecules. 38. The MNDO
    Method. Approximations and Parameters, J. Am. Chem. Soc. 1977,
    99, 4899–4907.

    Husch, T., Vaucher, A. C., & Reiher, M. (2018). Semiempirical molecular
    orbital models based on the neglect of diatomic differential overlap
    approximation. International journal of quantum chemistry, 118(24), e25799.

    James Stewart's MOPAC. http://openmopac.net/manual/nddo2e2c.html

"""
import json
from math import factorial
import numpy as np

# Load relevant files (MNDO parameters, ERIs multipoles, etc.)

with open('./SYSTEMS/mndo_params.json', 'r') as mndopar:
    PARAMS = json.load(mndopar)
with open('./SYSTEMS/element.json', 'r') as elems:
    ATOMS = json.load(elems)
with open('./SYSTEMS/multipole.json', 'r') as mpoles:
    MPOLES = json.load(mpoles)

# Multipole configuration


class MultiPole():
    """Contains the definition of the multipolar approximation
    used to compute the ERIs

    atom_a --> First center
    atom_b --> Second center
    zeta_mu --> Exponent of the STO for atom mu
    zeta_nu --> Exponent of the STO for atom nu
    n_mu --> Principal quantum number for the atom mu
    n_nu --> Principal quantum number for the atom nu
    d_dip_sp --> Distance between charges in the dipole
    d_qad_pp --> Distance between charges in the quadrupole

    """

    def __init__(self, atom_a, atom_b, bra=("1s", "1s"), ket=("1s", "1s")):
        self.bra = bra
        self.ket = ket
        self.atom_a = atom_a
        self.atom_b = atom_b
        self.zeta_mu = PARAMS["elements"][atom_a]["zs"]
        self.zeta_nu = PARAMS["elements"][atom_a]["zs"]
        self.n_mu = int(ATOMS["elements"][atom_a][3][0][0])
        self.n_nu = int(ATOMS["elements"][atom_b][3][0][0])
        self.d_dip_sp = self.__a_func(1) / np.sqrt(3)
        self.d_qad_pp = np.sqrt(self.__a_func(2)) / np.sqrt(5)

    def __a_func(self, a_par):
        """Function needed to compute dipole and quadrupole
        ditance of the charge arragement"""
        n_mu = int(self.bra[0][0])
        n_nu = int(self.bra[0][0])
        if "s" in self.bra[0]:
            zeta_mu =  PARAMS["elements"][self.atom_a]["zs"]
        elif "p" in self.bra[0]:
            zeta_mu =  PARAMS["elements"][self.atom_a]["zs"]
        if "s" in self.bra[1]:
            zeta_nu =  PARAMS["elements"][self.atom_a]["zs"]
        elif "p" in self.bra[1]:
            zeta_nu =  PARAMS["elements"][self.atom_a]["zp"]
        aval = ((2 * zeta_mu) ** (n_mu + 1 / 2) * (2 * zeta_nu) **
                (n_nu + 1 / 2) * (zeta_mu + zeta_nu) **
                (-n_mu - n_nu -  a_par - 1) * (factorial(2 * n_mu) *
                                               factorial(2 * n_nu)) **
                (-1 / 2) * factorial(n_mu + n_nu + a_par))
        return aval

    def __decode_dist(self, r_vect):
        """Substitutes the string in the multipoles' charge
        by the actual value of the distances"""
        for idx, coord in enumerate(r_vect):
            if coord == "Dmu":
                r_vect[idx] = self.d_dip_sp
            elif coord == "-Dmu":
                r_vect[idx] = -self.d_dip_sp
            elif coord == "Dqu":
                r_vect[idx] = self.d_qad_pp
            elif coord == "-Dqu":
                r_vect[idx] = -self.d_qad_pp
            elif coord == "2Dqu":
                r_vect[idx] = 2 * self.d_qad_pp
            elif coord == "-2Dqu":
                r_vect[idx] = -2 * self.d_qad_pp
        return r_vect

    def __get_rho(self, v_pho, atom):
        """Returns the pho coefficient corresponding to the charge
        distribution needed in the Klopman approximation"""
        pho = None
        if v_pho == "v_qss":
            pho = 1 / (2 * PARAMS["elements"][atom]["gss"])
        elif v_pho == "v_qpp":
            pho = 1 / (2 * PARAMS["elements"][atom]["gss"])
        elif v_pho == "v_musp":
            pho_0 = (PARAMS["elements"][atom]["hsp"] /
                     (self.d_dip_sp ** 2)) ** (1 / 3)
            pho_1 = 0
            while True:
                a_0 = 0.5 * pho_0 - 0.5 * (4 * self.d_dip_sp ** 2 +
                                           pho_0 ** -2) ** (-1 / 2)
                a_1 = 0.5 * pho_1 - 0.5 * (4 * self.d_dip_sp ** 2 +
                                           pho_1 ** -2) ** (-1 / 2)
                pho = pho_0 + (pho_1 - pho_0) * (
                    (PARAMS["elements"][atom]["hsp"] - a_0) /
                    (a_1 - a_0))
                if pho - pho_1 <= 1e-6:
                    break
                pho_1 = pho
                pho_0 = pho_1
        elif v_pho == "v_Qpp":
            pho_0 = (PARAMS["elements"][atom]["hsp"] /
                     (3 * self.d_qad_pp ** 4)) ** (1 / 5)
            pho_1 = 0
            while True:
                a_0 = 0.25 * pho_0 - 0.5 * (4 * self.d_qad_pp ** 2 +
                                            pho_0 ** -2) ** (-1 / 2) +\
                0.25 *(8 * self.d_qad_pp ** 2 + pho_0 ** -2) ** (-1 / 2)
                a_1 = 0.25 * pho_1 - 0.5 * (4 * self.d_qad_pp ** 2 +
                                            pho_1 ** -2) ** (-1 / 2) +\
                0.25 *(8 * self.d_qad_pp ** 2 + pho_1 ** -2) ** (-1 / 2)
                pho = pho_0 + (pho_1 - pho_0) * (
                    (PARAMS["elements"][atom]["hsp"] - a_0) /
                    (a_1 - a_0))
                if pho - pho_1 <= 1e-6:
                    break
                pho_1 = pho
                pho_0 = pho_1
        return pho

    def klopman(self, pol_bra, pol_ket, pho_t, pho_s):
        """Calculates the electrostatic multipolar energy
        using the Klopman approximation"""
        u_mnls = 0
        for pol_t in MPOLES["charge_posi"][pol_bra]:
            for pol_s in MPOLES["charge_posi"][pol_ket]:
                r_t = self.__decode_dist(pol_t[::-1])
                r_s = self.__decode_dist(pol_s[::-1])
                u_mnls += pol_t[-1] * pol_s[-1] /\
                    np.sqrt(np.linalg.norm(r_t - r_s) ** 2 +
                            (pho_t + pho_s) ** 2)

    def eval_eri(self):
        """Evaluates a given ERI using the Klopman approximation
        bra --> charge density corresponding to atom_a
        ket --> charge density corresponding to atom_b
        """
        eri = 0
        for charg_bra in MPOLES["charge_dist"][self.bra][0]:
            for charg_ket in MPOLES["charge_dist"][self.ket][0]:
                eri += self.klopman(charg_bra, charg_ket)
        return eri
