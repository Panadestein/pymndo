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
import scipy.constants as sc

# Define constants

EVAU = sc.physical_constants['atomic unit of electric potential'][0]

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

    """

    def __init__(self, atom_a, atom_b, bra=("1s", "1s"), ket=("1s", "1s")):
        self.bra = bra
        self.ket = ket
        self.atom_a = atom_a
        self.atom_b = atom_b

    def klopman(self, pol_bra, pol_ket, pho_bra, pho_ket):
        """Calculates the electrostatic multipolar energy
        using the Klopman approximation"""
        u_mnls = 0
        for pol_t in MPOLES["charge_posi"][pol_bra]:
            for pol_s in MPOLES["charge_posi"][pol_ket]:
                r_t = np.array(decode_dist(pol_t[::-1], self.bra, self.atom_a))
                r_s = np.array(decode_dist(pol_s[::-1], self.ket, self.atom_b))
                u_mnls += pol_t[-1] * pol_s[-1] /\
                    np.sqrt(np.linalg.norm(r_t - r_s) ** 2 +
                            (pho_bra + pho_ket) ** 2)
        return u_mnls

    def eval_eri(self):
        """Evaluates a given ERI using the Klopman approximation
        bra --> charge density corresponding to atom_a
        ket --> charge density corresponding to atom_b
        """
        eri = 0
        dens_bra = self.bra[0][1:] + self.bra[1][1:]
        dens_ket = self.ket[0][1:] + self.ket[1][1:]
        for idx_bra, charg_bra in enumerate(MPOLES["charge_dist"][dens_bra][0]):
            for idx_ket, charg_ket in enumerate(MPOLES["charge_dist"][dens_ket][0]):
                pho_bra = get_rho(MPOLES["charge_dist"][dens_bra][1][idx_bra],
                                  self.bra, self.atom_a)
                pho_ket = get_rho(MPOLES["charge_dist"][dens_ket][1][idx_ket],
                                  self.ket, self.atom_b)
                eri += self.klopman(charg_bra, charg_ket, pho_bra, pho_ket)
        eri = EVAU * eri
        return eri


def a_func(bra, atom, a_par):
    """Function needed to compute dipole and quadrupole
    ditance of the charge arragement

    zeta_mu --> Exponent of the STO for atom mu
    zeta_nu --> Exponent of the STO for atom nu
    n_mu --> Principal quantum number for the atom mu
    n_nu --> Principal quantum number for the atom nu

    """
    n_mu = int(bra[0][0])
    n_nu = int(bra[1][0])

    atom = str(atom)

    if "s" in bra[0]:
        zeta_mu = PARAMS["elements"][atom]["zs"]
    elif "p" in bra[0]:
        zeta_mu = PARAMS["elements"][atom]["zs"]
    if "s" in bra[1]:
        zeta_nu = PARAMS["elements"][atom]["zs"]
    elif "p" in bra[1]:
        zeta_nu = PARAMS["elements"][atom]["zp"]

    aval = ((2 * zeta_mu) ** (n_mu + 1 / 2) * (2 * zeta_nu) **
            (n_nu + 1 / 2) * (zeta_mu + zeta_nu) **
            (-n_mu - n_nu -  a_par - 1) * (factorial(2 * n_mu) *
                                           factorial(2 * n_nu)) **
            (-1 / 2) * factorial(n_mu + n_nu + a_par))
    return aval


def decode_dist(r_vect, bra, atom):
    """Substitutes the string in the multipoles' charge
    by the actual value of the distances"""

    d_dip_sp = a_func(bra, atom, 1) / np.sqrt(3)
    d_qad_pp = np.sqrt(a_func(bra, atom, 2)) / np.sqrt(5)

    for idx, coord in enumerate(r_vect):
        if coord == "Dmu":
            r_vect[idx] = d_dip_sp
        elif coord == "-Dmu":
            r_vect[idx] = -d_dip_sp
        elif coord == "Dqu":
            r_vect[idx] = d_qad_pp
        elif coord == "-Dqu":
            r_vect[idx] = -d_qad_pp
        elif coord == "2Dqu":
            r_vect[idx] = 2 * d_qad_pp
        elif coord == "-2Dqu":
            r_vect[idx] = -2 * d_qad_pp
    return r_vect


def get_rho(v_pho, bra, atom):
    """Returns the pho coefficient corresponding to the charge
    distribution needed in the Klopman approximation"""

    d_dip_sp = a_func(bra, atom, 1) / np.sqrt(3)
    d_qad_pp = np.sqrt(a_func(bra, atom, 2)) / np.sqrt(5)

    atom = str(atom)

    pho = None
    if v_pho == "v_qss":
        pho = EVAU / (2 * PARAMS["elements"][atom]["gss"])

    elif v_pho == "v_qpp":
        pho = EVAU / (2 * PARAMS["elements"][atom]["gss"])

    elif v_pho == "v_musp":
        pho_0 = (PARAMS["elements"][atom]["hsp"] /
                 (EVAU * d_dip_sp ** 2)) ** (1 / 3)
        pho_1 = pho_0 + 0.1
        while True:
            a_0 = 0.5 * pho_0 - 0.5 * (4 * d_dip_sp ** 2 + pho_0 ** -2) ** -0.5
            a_1 = 0.5 * pho_1 - 0.5 * (4 * d_dip_sp ** 2 + pho_1 ** -2) ** -0.5
            pho = pho_0 + (pho_1 - pho_0) * (
                (PARAMS["elements"][atom]["hsp"] / EVAU - a_0) /
                (a_1 - a_0))
            if pho - pho_1 <= 1e-6:
                break
            pho_0 = pho_1
            pho_1 = pho

    elif v_pho == "v_Qpp":
        pho_0 = (PARAMS["elements"][atom]["hsp"] /
                 (EVAU * 3 * d_qad_pp ** 4)) ** (1 / 5)
        pho_1 = pho_0 + 0.1
        while True:
            a_0 = 0.25 * pho_0 - 0.5 * (4 * d_qad_pp ** 2 +
                                        pho_0 ** -2) ** (-1 / 2) +\
            0.25 * (8 * d_qad_pp ** 2 + pho_0 ** -2) ** (-1 / 2)
            a_1 = 0.25 * pho_1 - 0.5 * (4 * d_qad_pp ** 2 +
                                        pho_1 ** -2) ** (-1 / 2) +\
            0.25 * (8 * d_qad_pp ** 2 + pho_1 ** -2) ** (-1 / 2)
            pho = pho_0 + (pho_1 - pho_0) * (
                (PARAMS["elements"][atom]["hsp"] / EVAU - a_0) /
                (a_1 - a_0))
            if pho - pho_1 <= 1e-6:
                break
            pho_0 = pho_1
            pho_1 = pho

    return pho


# Test with H2

if __name__ == "__main__":
    H2MPOLE = MultiPole("6", "1", ("2py", "2pz"))
    print(H2MPOLE.eval_eri())
