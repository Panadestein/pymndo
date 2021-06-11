"""This module computes the overlap matrix of Slater Type Orbitals
approximated by CGTOs.

RF Stewart, JCP 52, 431 (1970)
Helgaker, Trygve, and Peter R. Taylor. Modern Electronic Structure (1995).
James Stewart's MOPAC (http://openmopac.net/Manual/Overlap_integrals.html)

"""
import json
from functools import reduce
import numpy as np

# Import GTO expansion coefficients and exponents

with open('./SYSTEMS/sto_6g.json', 'r') as sto6g:
    STO6G = json.load(sto6g)
with open('./SYSTEMS/element.json', 'r') as elems:
    ATOMS = json.load(elems)
with open('./SYSTEMS/mndo_params.json', 'r') as mndopar:
    PARAMS = json.load(mndopar)

# Class containing a contracted GTOs expansion of the STOs


class ContrGTO():
    """Contains the definition of a GTO expansion for an
    STO-6G. Makes use of the properties of Hermite Gaussians"""
    def __init__(self, exps, coeff, origin=(0., 0., 0.), shell=(0, 0, 0)):
        self.origin = np.asarray(origin)
        self.shell = shell
        self.exps = exps
        self.coeff = coeff
        self.norm = None
        self.normalize()

    def normalize(self):
        """Normalizes the basis functions"""
        l_i, m_i, n_i = self.shell
        l_total = sum(self.shell)

        # Normalize PGBFs
        self.norm = np.sqrt(2 ** (2 * l_total + 1.5) * np.power(self.exps,
                                                                l_total + 1.5)/
                            (dfact(2 * l_i - 1) * dfact(2 * m_i - 1) *
                             dfact(2 * n_i - 1) * np.power(np.pi, 1.5)))

        # Normalize CGBFs
        n_fact = 0.0
        n_exps = range(len(self.exps))
        coeff_n = np.power(np.pi, 1.5) * dfact(2 * l_i - 1) *\
            dfact(2 * m_i - 1) * dfact(2 * n_i - 1) / 2 ** l_total

        for idx_a in n_exps:
            for idx_b in n_exps:
                n_fact += self.norm[idx_a] * self.norm[idx_b] *\
                        self.coeff[idx_a] * self.coeff[idx_b] /\
                        np.power(self.exps[idx_a] + self.exps[idx_b],
                                 l_total + 1.5)

        n_fact *= coeff_n
        n_fact = np.power(n_fact, -0.5)

        for idx_a in n_exps:
            self.coeff[idx_a] *= n_fact

# Compute overlap between two CGTOs


def dfact(n_num):
    """Double factorial"""
    if n_num == -1:
        return 1
    return reduce(int.__mul__, range(n_num, 0, -2))


def get_shell(orb):
    """Get the angular quantum numbers for a GTO
    according to its respective shell"""
    if "s" in orb:
        return (0, 0, 0)
    if "px" in orb:
        return (1, 0, 0)
    if "py" in orb:
        return (0, 1, 0)
    if "pz" in orb:
        return (0, 0, 1)
    return None


def e_ij_t(a_mo, b_mo, q_dist, zeta_a, zeta_b, t_coeff):
    """Recursive definition of Hermite Gaussian coefficients"""
    p_coeff = zeta_a + zeta_b
    q_coeff = zeta_a * zeta_b / p_coeff

    if (t_coeff < 0) or (t_coeff > (a_mo + b_mo)):
        return 0.0
    if a_mo == b_mo == t_coeff == 0:
        return np.exp(-q_coeff * q_dist * q_dist)
    if b_mo == 0:
        return (1 / (2 * p_coeff)) * e_ij_t(a_mo - 1, b_mo, q_dist, zeta_a,
                                            zeta_b, t_coeff - 1) -\
                (q_coeff * q_dist / zeta_a) * e_ij_t(a_mo - 1, b_mo, q_dist, zeta_a,
                                                     zeta_b, t_coeff) +\
                (t_coeff + 1) * e_ij_t(a_mo - 1, b_mo, q_dist, zeta_a,
                                       zeta_b, t_coeff + 1)
    return (1 / (2 * p_coeff)) * e_ij_t(a_mo, b_mo - 1, q_dist, zeta_a,
                                        zeta_b, t_coeff - 1) -\
            (q_coeff * q_dist / zeta_a) * e_ij_t(a_mo, b_mo - 1, q_dist, zeta_a,
                                                 zeta_b, t_coeff) +\
            (t_coeff + 1) * e_ij_t(a_mo, b_mo - 1, q_dist, zeta_a,
                                   zeta_b, t_coeff + 1)


def gau_overlap(zeta_a, t_a_mo, a_pos, zeta_b, t_b_mo, b_pos):
    """Evaluates the overlap between two primitive gaussians"""
    l_a, m_a, n_a = t_a_mo
    l_b, m_b, n_b = t_b_mo
    s_x = e_ij_t(l_a, l_b, a_pos[0] - b_pos[0], zeta_a, zeta_b, 0)
    s_y = e_ij_t(m_a, m_b, a_pos[1] - b_pos[1], zeta_a, zeta_b, 0)
    s_z = e_ij_t(n_a, n_b, a_pos[2] - b_pos[2], zeta_a, zeta_b, 0)

    return s_x * s_y * s_z * np.power(np.pi / (zeta_a + zeta_b), 1.5)


def s_ij(gau_a, gau_b):
    """Evaluates the overlap between two contracted gaussians"""
    s_val = 0
    for idx_a, coeff_a in enumerate(gau_a.coeff):
        for idx_b, coeff_b in enumerate(gau_b.coeff):
            s_val += gau_a.norm[idx_a] * gau_b.norm[idx_b] * coeff_a *\
                    coeff_b * gau_overlap(gau_a.exps[idx_a], gau_a.shell,
                                          gau_a.origin, gau_b.exps[idx_b],
                                          gau_b.shell, gau_b.origin)
    return s_val


# Return the overlap matrix for a given system

def s_matrix(molecule):
    """Returns the overlap matrix for the molecule of
    interest under the MNDO approximation. The matrix
    is written in the same order as the input atoms

    molecule --> Molecule() object

    """
    row = []
    for aidx, atom in enumerate(molecule.atoms):
        chi = [str(atom) + "_" + spo + "_" + str(aidx)
               for spo in ATOMS["elements"][str(atom)][3]]
        row.extend(chi)

    s_mat = np.zeros((len(row), len(row)))
    for idx_a, chi_a in enumerate(row):
        for idx_b, chi_b in enumerate(row):
            if "s" in chi_a:
                zeta_a = PARAMS["elements"][chi_a.split("_")[0]]["zs"]
            else:
                zeta_a = PARAMS["elements"][chi_a.split("_")[0]]["zp"]
            coeff_a = np.array(STO6G[chi_a.split("_")[1][:2]]["coeff"])
            exp_a = np.array(STO6G[chi_a.split("_")[1][:2]]["exp"]) * zeta_a
            origin_a = molecule.coords[int(chi_a.split("_")[2])]
            shell_a = get_shell(chi_a.split("_")[1])

            if "s" in chi_b:
                zeta_b = PARAMS["elements"][chi_b.split("_")[0]]["zs"]
            else:
                zeta_b = PARAMS["elements"][chi_b.split("_")[0]]["zp"]
            coeff_b = np.array(STO6G[chi_b.split("_")[1][:2]]["coeff"])
            exp_b = np.array(STO6G[chi_b.split("_")[1][:2]]["exp"]) * zeta_b
            origin_b = molecule.coords[int(chi_b.split("_")[2])]
            shell_b = get_shell(chi_b.split("_")[1])

            gau_a = ContrGTO(exp_a, coeff_a, origin_a, shell_a)
            gau_b = ContrGTO(exp_b, coeff_b, origin_b, shell_b)
            s_mat[idx_a, idx_b] = s_ij(gau_a, gau_b)

    return s_mat


if __name__ == '__main__':
    # Test with the H2O molecule
    from molecule import Molecule
    H2O_MOL = Molecule('H2O', [(8, (0.00000000, 0.00000000, 0.04851804)),
                               (1, (0.75300223, 0.00000000, -0.51923377)),
                               (1, (-0.75300223, 0.00000000, -0.51923377))],
                       charge=1, multiplicity=2)
    print(s_matrix(H2O_MOL))
