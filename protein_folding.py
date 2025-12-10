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

def hydro_forces(
        positions,
        keys,
        masses,
        residue_names,
        hydroph_m,
        cutoff=5.0,
        min_exclusion=2.0
    ):
    """
    Find non-bonded, non-adjacent contacts filtered by hydrophobicity,
    and compute forces using hydrophobicity 'm' values:

        F = 2*(m_i + m_j)/r^3 * rhat

    EXCLUDES:
      • Adjacent residues
      • Too-close residues (d < min_exclusion)
      • Pairs where BOTH hydrophobicity m-values = 0

    Parameters
    ----------
    positions : (N,3) array
    keys : list of (chain, resseq)
    masses : (N,) array (kept but not used for hydrophobic forces)
    residue_names : list of 3-letter residue names aligned to positions
    hydroph_m : dict mapping 3-letter codes → m values
    cutoff : float (upper cutoff)
    min_exclusion : float (lower cutoff)

    Returns
    -------
    contacts : list of ((chain_i, resseq_i), (chain_j, resseq_j), distance)
    forces   : (N,3) accumulated force vectors
    """

    N = len(positions)
    forces = np.zeros_like(positions)

    # Distance matrix
    diff = positions[:, None, :] - positions[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    contacts = []

    for i in range(N):
        for j in range(i+1, N):

            chain_i, res_i = keys[i]
            chain_j, res_j = keys[j]

            # ===== 1. EXCLUDE adjacent residues =====
            if chain_i == chain_j and abs(res_i - res_j) == 1:
                continue

            # ===== 2. Hydrophobicity lookup =====
            aa_i = residue_names[i]
            aa_j = residue_names[j]

            m_i = hydroph_m.get(aa_i, 0.0)
            m_j = hydroph_m.get(aa_j, 0.0)

            # ===== 3. EXCLUDE pairs where both hydrophobicities are zero =====
            if m_i == 0.0 and m_j == 0.0:
                continue

            d = dist_matrix[i, j]

            # ===== 4. EXCLUDE pairs that are too close =====
            if d < min_exclusion:
                continue

            # ===== 5. INCLUDE only pairs within cutoff =====
            if d < cutoff:
                contacts.append((keys[i], keys[j], d))

                # compute force
                rij = diff[i, j]
                rhat = rij / d
                fmag = 2 * (m_i + m_j) / (d**3)

                fij = fmag * rhat
                forces[i] += fij
                forces[j] -= fij

    return contacts, forces

'''turns coordinates into pairwise bond vectors, with magnitude
rij between residue i & consecutive residue j of size N-1'''
def bond_vectors(coords):
    return coords[1:, :] - coords[:-1, :]
""" calculates euclidean norm of each bond and returns magnitudes,
  i.e. the bond lengths
"""
def bond_lengths(coords):
    rij = bond_vectors(coords)
    return np.linalg.norm(rij, axis = 1)
#sets up arra of the angle triple indices, i.e. [(0,1,2),(1,2,3),(2,3,4),...]
#allows for vectorization of code in functions below
def angle_triplets (N_residues):

    triplets = np.column_stack([
    np.arange(N_residues-2),      # i
    np.arange(1, N_residues-1),   # j (vertex)
    np.arange(2, N_residues)      # k
])
    return triplets
#sets up arra of the angle quadruple indices, i.e. [(0,1,2,3),(1,2,3,4),(2,3,4,5),...]
#allows for vectorization of code in functions below
def angle_quadruples (N_residues):

    quadruples = np.column_stack([
    np.arange(N_residues-3),      
    np.arange(1, N_residues-2),   
    np.arange(2, N_residues-1),   
    np.arange(3, N_residues)
])
    return quadruples

def planar_bond_angles(positions, angle_triplets, degrees=True):
    """
    positions      : (N,3)
    angle_triplets : (N-2,3) integer indices (i,j,k) meaning angle i-j-k
    """
    a = positions[angle_triplets[:,0]]
    b = positions[angle_triplets[:,1]]
    c = positions[angle_triplets[:,2]]

    u = a - b                 # (M,3)
    v = c - b                 # (M,3)

    dot = np.sum(u * v, axis=1)
    nu  = np.linalg.norm(u, axis=1)
    nv  = np.linalg.norm(v, axis=1)

    cos_theta = dot / (nu * nv)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))

    return np.degrees(theta) if degrees else theta

def dihedral_angles(positions, angle_quadruples, degrees=True):
    """
    Vectorized signed dihedral (torsion) angles for quadruples (a,b,c,d).

    positions        : (N,3)
    angle_quadruples : (N-3,4) integer array of atom indices (a,b,c,d)

    Returns:
        (N-3,) dihedral angles in degrees.
    """

    a = positions[angle_quadruples[:,0]]
    b = positions[angle_quadruples[:,1]]
    c = positions[angle_quadruples[:,2]]
    d = positions[angle_quadruples[:,3]]

    # Bond vectors
    b1 = b - a             # (M,3)
    b2 = c - b             # (M,3)
    b3 = d - c             # (M,3)

    # Normals to the planes
    n1 = np.cross(b1, b2)  # (M,3)
    n2 = np.cross(b2, b3)  # (M,3)

    # Norms
    n1_norm = np.linalg.norm(n1, axis=1)
    n2_norm = np.linalg.norm(n2, axis=1)
    b2_norm = np.linalg.norm(b2, axis=1)

    # Avoid division by zero (invalid dihedrals)
    mask = (n1_norm < 1e-12) | (n2_norm < 1e-12) | (b2_norm < 1e-12)

    # Unit normals
    n1u = n1 / n1_norm[:,None]
    n2u = n2 / n2_norm[:,None]
    b2u = b2 / b2_norm[:,None]

    # m = n1 × b2_unit
    m = np.cross(n1u, b2u)

    # Components for atan2
    x = np.sum(n1u * n2u, axis=1)    # dot(n1u, n2u)
    y = np.sum(m * n2u, axis=1)      # dot(m,   n2u)

    phi = np.arctan2(y, x)           # signed dihedral

    # Convert to degrees if needed
    if degrees:
        phi = np.degrees(phi)

    # Assign NaN to invalid angles
    phi[mask] = np.nan

    return phi

#Below are functions for the various forces which arise from contact & non_conttact potentials
#in addition to the external constant-velocity pulling introduced to unfold our protein

#Intramolecular forces between our amino acids, modeled as spring like
#using hooke's law with experimentally determined 'spring constant' K_r
def F_bonds(positions, k_r, r_eq):
    """
    Harmonic bond forces for a linear polymer chain.

    positions : (N,3)
    k_r       : scalar   force 'spring-like' constant
    r_eq      : (N-1,) or scalar  - equilibrium bond lengths

    Returns:
        F : (N,3) forces on each atom
    """

    # Bond vectors: r_{i+1} - r_i
    rij = positions[1:] - positions[:-1]         # (N-1,3)

    # Bond lengths
    r = np.linalg.norm(rij, axis=1)              # (N-1,)
    rhat = rij / r[:, None]                      # normalized bond vectors

    # Force magnitudes  (scalar stretch) 
    # F = - d/dr [ k (r - r_eq)^2 ] = -2 k (r - r_eq)
    fmag = -2.0 * k_r * (r - r_eq)               # (N-1,)

    # Vector form of bond forces
    forces = fmag[:, None] * rhat                # (N-1,3)

    # Scatter to amino acids
    N = len(positions)
    F = np.zeros((N,3))
    np.add.at(F, np.arange(N-1),  forces)
    np.add.at(F, np.arange(1, N), -forces)

    return F

def F_bb_angles(positions, angle_triples, k_theta, theta_eq):
    """
    positions : (N,3)
    angles    : (A,3) angle triples (i,j,k)
    k_theta   : scalar or (A,) force constants
    theta_eq  : scalar or (A,) equilibrium angle
    """

    i = angle_triples[:,0]
    j = angle_triples[:,1]
    k = angle_triples[:,2]

    # Vectors from central atom j → i and j → k
    rij = positions[i] - positions[j]     # (A,3)
    rkj = positions[k] - positions[j]     # (A,3)

    # Norms
    rij_norm = np.linalg.norm(rij, axis=1)
    rkj_norm = np.linalg.norm(rkj, axis=1)

    # Unit vectors
    e_ij = rij / rij_norm[:,None]
    e_kj = rkj / rkj_norm[:,None]

    # Angle cosine
    cos_theta = np.sum(e_ij * e_kj, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Angle
    theta = np.arccos(cos_theta)

    # dU/dθ for harmonic angle potential
    dU_dtheta = 2.0 * k_theta * (theta - theta_eq)

    # sin(theta) with stability fix
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    sin_theta = np.where(sin_theta < 1e-12, 1e-12, sin_theta)

    # Standard MD angle force formulas (OpenMM/GROMACS/CHARMM)
    n_i = (cos_theta[:,None] * e_ij - e_kj) / (rij_norm * sin_theta)[:,None]
    n_k = (cos_theta[:,None] * e_kj - e_ij) / (rkj_norm * sin_theta)[:,None]

    Fi = -dU_dtheta[:,None] * n_i
    Fk = -dU_dtheta[:,None] * n_k
    Fj = -(Fi + Fk)

    # Scatter forces
    F = np.zeros_like(positions)
    np.add.at(F, i, Fi)
    np.add.at(F, j, Fj)
    np.add.at(F, k, Fk)

    return F


def F_dihedrals(positions, dihedral_quads, k_phi, phi_eq):
    """
    positions      : (N,3)
    dihedral_quads : (D,4) array of (i,j,k,l)
    k_phi          : scalar or (D,) force constant
    phi_eq         : scalar or (D,) equilibrium dihedral angle
    """

    i = dihedral_quads[:,0]
    j = dihedral_quads[:,1]
    k = dihedral_quads[:,2]
    l = dihedral_quads[:,3]

    # Bond vectors
    b1 = positions[i] - positions[j]        # (D,3)
    b2 = positions[k] - positions[j]        # (D,3)
    b3 = positions[l] - positions[k]        # (D,3)

    # Normals to planes
    n1 = np.cross(b1, b2)                   # (D,3)
    n2 = np.cross(b2, b3)                   # (D,3)

    n1_norm = np.linalg.norm(n1, axis=1)
    n2_norm = np.linalg.norm(n2, axis=1)
    b2_norm = np.linalg.norm(b2, axis=1)

    # Avoid degeneracy
    mask = (n1_norm < 1e-12) | (n2_norm < 1e-12) | (b2_norm < 1e-12)

    n1u = n1 / n1_norm[:,None]
    n2u = n2 / n2_norm[:,None]
    b2u = b2 / b2_norm[:,None]

    # Compute dihedral angle phi
    m = np.cross(n1u, b2u)
    x = np.sum(n1u * n2u, axis=1)
    y = np.sum(m * n2u, axis=1)
    phi = np.arctan2(y, x)

    # dU/dφ for harmonic torsion
    dU_dphi = 2.0 * k_phi * (phi - phi_eq)

    # Needed geometric factors
    inv_n1 = 1.0 / n1_norm
    inv_n2 = 1.0 / n2_norm
    inv_b2 = 1.0 / b2_norm

    # Force components (gold-standard MD formula)
    # These gradients come from ∂φ/∂ri etc.
    t1 =  (n1 * inv_n1[:,None]) * inv_b2[:,None]
    t2 =  (n2 * inv_n2[:,None]) * inv_b2[:,None]

    Fi = -dU_dphi[:,None] * t1
    Fl =  dU_dphi[:,None] * t2
    Fj = -(Fi + np.cross(b2u, Fi) * b2_norm[:,None])
    Fk = -(Fl + np.cross(b2u, Fl) * b2_norm[:,None])

    # Scatter
    F = np.zeros_like(positions)
    np.add.at(F, i, Fi)
    np.add.at(F, j, Fj)
    np.add.at(F, k, Fk)
    np.add.at(F, l, Fl)

    return F

#External pulling force (Optical Tweezer) We can choose to pull in just one direction,
#i.e. setting y & z velocity components to zero & choosing an x-only velocity value
def F_pull(positions, t, idx, k_trap, r_trap0, v_pull):
    """
    positions : (N,3)
    t         : scalar time
    idx       : index of pulled bead (usually last one, so just -1, but here for generalizability )(int)
    k_trap    : trap stiffness (scalar)
    r_trap0   : (3,) initial trap center position
    v_pull    : (3,) pulling velocity vector
    """

    # Trap center at time t
    r_trap = r_trap0 + v_pull * t          # (3,)

    # Vector from trap center to bead
    delta = positions[idx] - r_trap        # (3,)

    # Spring force
    F = np.zeros_like(positions)
    F[idx] = -k_trap * delta               # (3,)

    return F


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

  
    
def total_force_contribution(
        positions, t,
        idx_pull, k_trap, r_trap0, v_pull,
        planar_triples, k_theta, theta_eq,
        dihedral_quads, k_phi, phi_eq,
        k_r, r_eq):
    """
    Compute total force at time t from:
    - bonds
    - angle bending
    - dihedral torsion
    - optical trap pulling

    All force components return an array of shape (N, 3)
    matching 'positions'.
    """

    F  = F_bonds(positions, k_r, r_eq)
    F += F_bb_angles(positions, planar_triples, k_theta, theta_eq)
    F += F_dihedrals(positions, dihedral_quads, k_phi, phi_eq)
    F += F_pull(positions, t, idx_pull, k_trap, r_trap0, v_pull)

    return F

def langevin_step(positions, velocities, masses, gamma, dt, kB, T, force_fn, t):
    """
    positions : (N,3)
    velocities: (N,3)
    masses    : (N,)
    gamma     : (N,)
    dt        : scalar timestep
    kB        : Boltzmann constant
    T         : Temperature
    force_fn  : function that computes total forces F(positions, t)
    t         : current time

    Returns updated (positions, velocities)
    """

    m = masses[:,None]       # (N,1)
    g = gamma[:,None]        # (N,1)

    # --- 1. Compute deterministic forces at time t ---
    F = force_fn(positions, t)   # (N,3)

    # --- 2. First noise term ---
    sigma = np.sqrt(2 * g * kB * T * dt)   # (N,1)
    W1 = sigma * np.random.normal(size = positions.shape)

    # --- 3. First half-step velocity update ---
    v_half = velocities + dt/(2*m) * (F - g*velocities + W1)

    # --- 4. Position update ---
    positions_new = positions + dt * v_half

    # --- 5. Forces at new positions ---
    F_new = force_fn(positions_new, t + dt)

    # --- 6. Second noise term ---
    W2 = sigma * np.random.normal(size = positions.shape)

    # --- 7. Second half velocity update ---
    velocities_new = v_half + dt/(2*m) * (F_new - g*v_half + W2)

    return positions_new, velocities_new
    
#Initialize system and iterate dynamics
#extract & set up masses
#initializa velocities
#set up parameters: gamma, T, r_eq, K_r, k_theta, ...
#choose dt that is both simulation & experiment appropriate

time_steps = tot_duration/dt
time = 0
for i in range(time_steps):
  F = total_force_contribution()
  positions, velocities = langevin_step(F,...)
  planar_angles = planar_bond_angles()
  dihedral_angles = dihedral_angles()

  time += dt


        
        
