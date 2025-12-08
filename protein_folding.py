import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import numpy as np

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "1NCT.pdb")

# store Cα atoms in a simple dict: (chain, resseq) → np.array([x,y,z])
ca_positions = {}

#hydrophobicity constants for applicable amino acids
hydroph_m = {'ARG': 0.3,
             'MET': 0.4,
             'LYS': 0.4,
             'VAL': 0.6,
             'ILE': 0.8,
             'LEU': 0.8,
             'PRO': 0.8,
             'TYR': 1.1,
             'PHE': 1.6,
             'TRP': 1.6,}
#maps 3 letter aa codes to 1 letter
aa3_to_aa1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",}
  

for model in structure:
    for chain in model:
        for residue in chain:
            if "CA" in residue:  # only alpha-carbon
                ca_positions[(chain.id, residue.id[1])] = residue["CA"].coord
print(ca_positions)

#convert dict with Ca positions to pure Numpy coordinate array
keys = sorted(ca_positions.keys())
coords = np.array([ca_positions[k] for k in keys])

def bond_lengths(coords):
    return np.linalg.norm(coords, axis = 1)
    
def angle_between_three(a, b, c, degrees=True):

    u = a - b
    v = c - b

    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)

    if nu == 0 or nv == 0:
        return np.nan

    cos_theta =  np.dot(u,v) / (nu * nv)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    return np.degrees(theta) if degrees else theta 

def dihedral(a, b, c, d, degrees=True):
    """
    Signed dihedral (torsion) angle for four points a-b-c-d, in degrees (default).
    Uses stable atan2-based formulation.
    (Find paper for dihedral angle calculation)
    """
    b1 = b - a
    b2 = c - b
    b3 = d - c

    # normals to planes
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    b2_norm = np.linalg.norm(b2)
    if n1_norm < 1e-12 or n2_norm < 1e-12 or b2_norm < 1e-12:
        return np.nan

    n1u = n1 / n1_norm
    n2u = n2 / n2_norm
    b2u = b2 / b2_norm

    m = np.cross(n1u, b2u)
    x = np.dot(n1u, n2u)
    y = np.dot(m, n2u)
    phi = np.arctan2(y, x)
    return np.degrees(phi) if degrees else phi


def backbone_bond_angles(coords, degrees = True):

    N = len(coords)
    angles = np.full(N, np.nan)

    for i in range(1, N-1):
        angles[i] = angle_between_three(angles[i - 1], angles[i], angles[i+1])
    return angles

def backbone_dihedral_angles(coords, degrees=True):

   N = len(coords)
   dihs = np.full(N, np.nan)
   for i in range(2, N-1):
        dihs[i] = dihedral(coords[i-2], coords[i-1], coords[i], coords[i+1], degrees=degrees)
    
   return dihs 

def F_bonds(positions, bonded_pairs, k_r, r_eq):
    
    """
    positions : (N,3)
    bonds     : (M,2) integer array of bonded pairs
    k_r       : (M,) force constants
    r_eq      : (M,) equilibrium bond lengths
    """

    # Vectorized coordinate differences
    rij = positions[bonded_pairs[:, 0]] - positions[bonded_pairs[:, 1]]  # (M,3)

    # Bond lengths
    r = np.linalg.norm(rij, axis=1)                         # (M,)
    rhat = rij / r[:, None]                                 # (M,3)

    # Force magnitude
    fmag = -2 * k_r * (r - r_eq)                            # (M,)

    # Vector forces
    forces = fmag[:, None] * rhat                           # (M,3)

    # Scatter forces to atoms
    F = np.zeros_like(positions)
    np.add.at(F, bonded_pairs[:, 0],  forces)
    np.add.at(F, bonded_pairs[:, 1], -forces)
    return F



    return

def F_bb_angles():
    # angles : (A,3) array of (i,j,k) defining angle i-j-k

    i, j, k = angles[:,0], angles[:,1], angles[:,2]

    rij = positions[i] - positions[j]
    rkj = positions[k] - positions[j]

    rij_norm = np.linalg.norm(rij, axis=1)
    rkj_norm = np.linalg.norm(rkj, axis=1)

    e_ij = rij / rij_norm[:,None]
    e_kj = rkj / rkj_norm[:,None]

    cos_theta = np.sum(e_ij * e_kj, axis=1)
    theta = np.arccos(cos_theta)

    # dU/dθ
    dU_dtheta = 2 * k_theta * (theta - theta_eq)

    sin_theta = np.sqrt(1 - cos_theta**2)
    sin_theta = np.where(sin_theta < 1e-12, 1e-12, sin_theta)

    # vector parts used by every MD code
    n_i = (cos_theta[:,None] * e_ij - e_kj) / (rij_norm * sin_theta)[:,None]
    n_k = (cos_theta[:,None] * e_kj - e_ij) / (rkj_norm * sin_theta)[:,None]

    Fi = -dU_dtheta[:,None] * n_i
    Fk = -dU_dtheta[:,None] * n_k
    Fj = -(Fi + Fk)

    F = np.zeros_like(positions)
    np.add.at(F, i, Fi)
    np.add.at(F, j, Fj)
    np.add.at(F, k, Fk)
    return F



    return 

def F_dih_angles():

    return

def F_pull():

    return

# Non contact force contributions from Lennard-Jones & Coulombic (electrostatic) potentials
def non_contact_forces(positions, pairs, A, B, charges, epsilon):

  """
    pairs : (P,2) atom pairs to evaluate
    A,B   : LJ parameters per pair
    charges: array of charges
    """
  i, j = pairs[:,0], pairs[:,1]

        rij = positions[i] - positions[j]
        r = np.linalg.norm(rij, axis=1)
        rhat = rij / r[:,None]
    
        # LJ + Coulomb derivatives
        dU_dr = (
            -12*A / r**13
            + 6*B / r**7
            - (charges[i] * charges[j]) / (epsilon * r**2)
        )
    
        fmag = -dU_dr
        fij = fmag[:,None] * rhat
    
        F = np.zeros_like(positions)
        np.add.at(F, i,  fij)
        np.add.at(F, j, -fij)
        return F

  


def run_langevin_dynamics(masses, F_pull):

    contact_potentials = contact_potentials(coords, bb_angles, dih_angles)
     
    



        
        
