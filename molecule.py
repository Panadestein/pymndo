"""
This module contains the definition of the Molecule class, which
contains molecular parameters like charge, multiplicity and geometry.
"""
import json
import numpy as np

with open('./SYSTEMS/element.json', 'r') as elems:
    ATOMS = json.load(elems)

class Molecule:
    """A simple class to hold molecular structure.
    Cartesian coordinates in Angstrom"""
    def __init__(self, name="MOL", geometry=None, charge=0, multiplicity=1):
        self.name = name
        self.charge = charge
        self.multiplicity = multiplicity
        self.geometry = geometry
        self.atoms = self.__get_atoms()
        self.coords = self.__get_coords()
    def __get_atoms(self):
        """Extract list of atoms from input geometry"""
        atms = []
        for z_coord in self.geometry:
            atms.append(z_coord[0])
        return atms
    def __get_coords(self):
        """Extract coordinates from input geometry"""
        coord = []
        for z_coord in self.geometry:
            coord.append(z_coord[1:])
        return np.array(coord)
    def get_amass(self):
        """Get the atomic masses of the atoms involved in the calculation"""
        mass = []
        for z_coord in self.geometry:
            mass.append(ATOMS['element'][str(z_coord[0])][1])
        return mass
    def get_zcore(self):
        """Returns the core nuclear charge of the atoms"""
        qcore = []
        for z_coord in self.geometry:
            qcore.append(ATOMS['element'][str(z_coord[0])][2])
        return qcore
