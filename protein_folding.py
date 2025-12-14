import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
import plotly.graph_objects as go
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "1NCT.pdb")

# store Cα atoms in a simple dict: (chain, resseq) → np.array([x,y,z])
ca_positions = {}

residue_names = []



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

charge_map = {
        'ASP': -1,
        'GLU': -1,
        'LYS': +1,
        'ARG': +1
    }
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
positions = np.array([ca_positions[k] for k in keys])
diff_first = positions[0]
positions -= diff_first #Shifts entire protein so first aa is at 0,0,0 and relative distances are same

for chain_id, resseq in keys:
    residue = structure[0][chain_id][resseq]   # Model 0
    residue_names.append(residue.resname)

def nb_forces(
        positions,
        keys,
        residue_names,
        hydroph_m,
        radius,
        N_A,
        charge_map,
    ):
    """
    Compute both hydrophobic and electrostatic forces.
    
    Hydrophobic force:
        F_h = 1.25 * (m_i + m_j) * (2*d - 2*radius) * (5.52 / N_A)

    Electrostatic force:
        F_e = 1.5 * q_i * q_j * (2*d - 2*radius) * (5.52 / N_A)

    Charges:
        ASP = -1
        GLU = -1
        LYS = +1
        ARG = +1
        others = 0
    """
    N = len(positions)
    forces = np.zeros_like(positions)

    # geometric cutoffs
    cutoff = np.sqrt(0.45) + 2 * radius
    min_exclusion = np.sqrt(0.37) + 2 * radius

    # distance matrix
    diff = positions[:, None, :] - positions[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    for i in range(N):
        for j in range(i+1, N):

            chain_i, res_i = keys[i]
            chain_j, res_j = keys[j]

            # skip adjacent residues
            if chain_i == chain_j and abs(res_i - res_j) == 1:
                continue

            aa_i = residue_names[i]
            aa_j = residue_names[j]

            # hydrophobicities
            m_i = hydroph_m.get(aa_i, 0.0)
            m_j = hydroph_m.get(aa_j, 0.0)

            # charges
            q_i = charge_map.get(aa_i, 0)
            q_j = charge_map.get(aa_j, 0)

            d = dist_matrix[i, j]

            # too close → skip
            if d < min_exclusion:
                continue

            if d < cutoff:

                rij = diff[i, j]
                rhat = rij / d

                # shared distance factor
                dist_term = (2*d - 2*radius)

                # --- Hydrophobic contribution ---
                F_h = 1.25 * (m_i + m_j) * dist_term * (5.52 / N_A)

                # --- Electrostatic contribution ---
                F_e = 1.5 * q_i * q_j * dist_term * (5.52 / N_A)

                # total magnitude
                fmag = F_h + F_e

                fij = fmag * rhat

                forces[i] += fij
                forces[j] -= fij

    return forces

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_protein_3d_interactive(positions, keys=None):
    """
    Plot a protein Cα trace in 3D using Plotly (interactive rotation).
    
    Parameters
    ----------
    positions : (N,3) numpy array
        Cα coordinates
    keys : list of tuples (chain, resseq)
        Optional labels for residues
    """

    # --- Backbone lines ---
    line_trace = go.Scatter3d(
        x=positions[:,0],
        y=positions[:,1],
        z=positions[:,2],
        mode='lines',
        line=dict(color='black', width=4),
        name='Backbone'
    )

    # --- Residue points ---
    points_trace = go.Scatter3d(
        x=positions[:,0],
        y=positions[:,1],
        z=positions[:,2],
        mode='markers+text' if keys is not None else 'markers',
        marker=dict(size=4, color='red'),
        text=[f"{c}{r}" for c,r in keys] if keys is not None else None,
        textposition="top center",
        name='Cα atoms'
    )

    fig = go.Figure(data=[line_trace, points_trace])
    fig.update_layout(
        title="Protein Cα Trace (3D Interactive)",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        )
    )
    fig.show(renderer="browser")

'''turns coordinates into pairwise bond vectors, with magnitude
rij between residue i & consecutive residue j of size N-1'''
def bond_vectors(positions):
    return positions[1:, :] - positions[:-1, :]
""" calculates euclidean norm of each bond and returns magnitudes,
  i.e. the bond lengths
"""
def bond_lengths(positions):
    rij = bond_vectors(positions)
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

def get_contour_length(positions):
    
    r_ij = bond_vectors(positions)
    r_ij_magnitudes = bond_lengths(r_ij)
    contour_length = np.sum(r_ij_magnitudes)

    return contour_length
    
def pulling_time(positions, v_pull, dt):
    contour_length = get_contour_length(positions)
    Tf = np.abs(2*contour_length/v_pull) #forward period, i.e. one forward pull cycle
    Tf = Tf
    return Tf

def v_pull_sawtooth(t, Tf, v_pull):
    """
    positions : (N,3)
    t         : scalar time (current time)
    x_min   : (3,) final bead being pulled
    v_pull    : (3,) pulling velocity vector
    x_max     :  (3,) user determined multiple of protein contour length
    """
    T_tot = 2 * Tf
    tau = t%T_tot

    if tau < Tf:
        return v_pull
    else:
        return -1 * v_pull 
    

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
    """
    N = len(positions)

    # Bond vectors
    rij = positions[1:] - positions[:-1]   # (N-1,3)

    # Bond lengths
    r = np.linalg.norm(rij, axis=1)        # (N-1,)

    # Protect against zero-length bonds
    eps = 1e-12
    r_safe = np.maximum(r, eps)

    rhat = rij / r_safe[:,None]            # (N-1,3)

    # Broadcast r_eq
    r_eq = np.asarray(r_eq)
    if r_eq.ndim == 0:
        r_eq = np.full_like(r, r_eq)

    # Force magnitude
    fmag = k_r * (r - r_eq)         # (N-1,)

    forces = fmag[:,None] * rhat           # (N-1,3)

    # Accumulate to atom forces
    F = np.zeros((N,3))
    F[:N-1] +=  forces
    F[1:   ] += -forces

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
    dU_dtheta = k_theta * (theta - theta_eq)

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
    dU_dphi = k_phi * (phi - phi_eq)

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
        positions,
        idx_pull, k_trap, r_trap0, v_pull,
        planar_triples, k_theta, theta_eq,
        dihedral_quads, k_phi, phi_eq,
        k_r, r_eq, keys, hydroph_m, radius, residue_names):
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
    F += nb_forces(positions, keys, residue_names, hydroph_m, radius, N_A, charge_map)
    #F += F_pull(positions, t, idx_pull, k_trap, r_trap0, v_pull)

    return F
          
def langevin_step(positions, gamma, dt, kB, T, total_force, N_a):
    """
    positions : (N,3)
    velocities: (N,3)
    masses    : (N,)
    gamma     : (N,)
    dt        : scalar timestep
    kB        : Boltzmann constant
    T         : Temperature
    force_fn  : function that computes total forces F(positions, t) (N,3)
    t         : current time

    Returns updated (positions, velocities)
    """

    #m = masses[:,None]       # (N,1)
    g = gamma[:,None]        # (N,1)

      # (N,3)

     # --- 2. First noise term ---
    sigma = np.sqrt((2 * (kB/N_a) * T * dt) / g)  # (N,1)
    W1 = sigma * np.random.normal(size = positions.shape)

    #----- 3. Half-step position update ---
    positions_new = positions + (dt/ g) * total_force + W1

    return positions_new

# ----System initialization and Dynamics Setup----
N = len(positions)

angle_triple_indices = angle_triplets(N)
angle_quadruple_indices = angle_quadruples(N)
positions = positions/10
#model parameters
radius = 0.2 # bead radii 
gamma = 5
# Change gamma to N,1 vector
gammas = np.full(N, gamma) 
#gammas = gammas[:,None]
dt = 0.3
kB = 0.0083
T = 300
N_a = 6e23

#equilibrium values
theta_eq = 110*(np.pi/180)
phi_eq = 130*(np.pi/180)
r_eq = 0.38
r_trap0 = 2


#stiffness factors
k_r = 500
k_phi = 2
k_theta = 30
k_trap = 1

tot_duration = 3e4
time_steps = tot_duration/dt
time = 0
idx_pull = -1 #pulling last bead in amino acid chain
r_trap0 = positions[idx_pull,:] 

initial_positions = positions.copy()
#print(initial_positions)
time_list = []
position_diff_list = []

N_A = 6.02214076e23
kB_per_particle = 1.380649e-26   # kJ / (K · molecule)
amu_to_kg = 1.66053906660e-27

def per_molecule(k_kj_per_mol):
    """Convert k (kJ/mol ...) -> kJ/(molecule ...)."""
    return k_kj_per_mol / N_A

def mass_to_md_units(mass_amu):
    """
    Convert mass in amu -> integrator mass units kJ·ps^2 / nm^2
    (1 kg = 1000 kJ·ps^2/nm^2 so multiply kg by 1000).
    """
    kg = mass_amu * amu_to_kg
    return kg * 1000.0  # kJ·ps^2 / nm^2

# --- Example: convert your stiffnesses once at initialization ---
k_r = per_molecule(500.0)      # kJ/(molecule·nm^2)
k_theta = per_molecule(30.0)   # kJ/(molecule·rad^2)
k_phi = per_molecule(2.0)      # kJ/molecule

# mass: scalar per bead
mass_amu = 110.0
m_md = mass_to_md_units(mass_amu)   # scalar in kJ·ps^2 / nm^2
masses_md = np.full(N, m_md)        # if N beads

# --- BAOAB integrator step ---
def baoab_step_pull(positions, velocities, masses_md, dt, gamma, T, F, F_pull, time, pull_period):
    """
    One BAOAB Langevin step.
    - positions: (N,3) nm
    - velocities: (N,3) nm/ps
    - masses_md: scalar or (N,) in kJ·ps^2/nm^2
    - dt: ps
    - gamma: 1/ps
    - T: Kelvin
    - compute_forces: function(positions) -> forces (N,3) in kJ/(molecule·nm)
    """
    # ensure array shapes
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    N = positions.shape[0]

    masses_md = np.asarray(masses_md, dtype=float)
    if masses_md.ndim == 0:
        masses_md = np.full(N, masses_md)

    inv_m = 1.0 / masses_md  # (N,) units: (nm^-2) / (kJ·ps^2) so F*inv_m -> nm/ps^2

    # Step B (half kick) : v += 0.5*dt * F / m  # kJ/(molecule·nm)
    # acceleration a = F * inv_m -> units nm/ps^2
    
    
    #comment out if code doesn't work
    #v_pull_st = v_pull_sawtooth(time, pull_period, v_pull)
    
    velocities = velocities + 0.5 * dt * (F * inv_m[:,None])
    

    # Step A (half drift): x += 0.5*dt * v
    positions = positions + 0.5 * dt * velocities

    # Step O (Ornstein-Uhlenbeck on velocities)
    # v -> c * v + sigma_v * Normal
    c = np.exp(-gamma * dt)
    # variance factor: (1 - c^2) * kB*T / m
    # note kB in kJ/(K·molecule), m in kJ·ps^2/nm^2 -> gives (nm/ps)^2
    sigma2 = (1.0 - c*c) * (kB_per_particle * T) * inv_m  # (N,) -> (nm/ps)^2
    # numeric safe-guard
    sigma2 = np.where(sigma2 < 0.0, 0.0, sigma2)
    sigma = np.sqrt(sigma2)  # (N,)
    # draw normal noise per particle & per dimension
    xi = np.random.normal(size=(N,3))
    velocities = (c * velocities) + (sigma[:,None] * xi)


    # Step A (half drift): x += 0.5*dt * v
    positions = positions + 0.5 * dt * velocities

    # Step B (half kick) with updated forces
    F = total_force_contribution(positions,
         idx_pull, k_trap, r_trap0, v_pull,
         angle_triple_indices, k_theta, theta_eq,
         angle_quadruple_indices, k_phi, phi_eq,
         k_r, r_eq, keys, hydroph_m, radius, residue_names)
    F[-1, 0] -= F_pull
    velocities = velocities + 0.5 * dt * (F * inv_m[:,None])
    
    


    ###IF PULL UNCOMMENT BELOW
    
    positions[0] = initial_positions[0]

    return positions, velocities

def baoab_step(positions, velocities, masses_md, dt, gamma, T, F, v_pull, time, pull_period):
    """
    One BAOAB Langevin step.
    - positions: (N,3) nm
    - velocities: (N,3) nm/ps
    - masses_md: scalar or (N,) in kJ·ps^2/nm^2
    - dt: ps
    - gamma: 1/ps
    - T: Kelvin
    - compute_forces: function(positions) -> forces (N,3) in kJ/(molecule·nm)
    """
    # ensure array shapes
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    N = positions.shape[0]

    masses_md = np.asarray(masses_md, dtype=float)
    if masses_md.ndim == 0:
        masses_md = np.full(N, masses_md)

    inv_m = 1.0 / masses_md  # (N,) units: (nm^-2) / (kJ·ps^2) so F*inv_m -> nm/ps^2

    # Step B (half kick) : v += 0.5*dt * F / m  # kJ/(molecule·nm)
    # acceleration a = F * inv_m -> units nm/ps^2
    
    
    #comment out if code doesn't work
    #v_pull_st = v_pull_sawtooth(time, pull_period, v_pull)
    
    velocities = velocities + 0.5 * dt * (F * inv_m[:,None])
    #velocities[-1] = np.array([v_pull_st, 0, 0]).reshape(1, 3)

    # Step A (half drift): x += 0.5*dt * v
    positions = positions + 0.5 * dt * velocities

    # Step O (Ornstein-Uhlenbeck on velocities)
    # v -> c * v + sigma_v * Normal
    c = np.exp(-gamma * dt)
    # variance factor: (1 - c^2) * kB*T / m
    # note kB in kJ/(K·molecule), m in kJ·ps^2/nm^2 -> gives (nm/ps)^2
    sigma2 = (1.0 - c*c) * (kB_per_particle * T) * inv_m  # (N,) -> (nm/ps)^2
    # numeric safe-guard
    sigma2 = np.where(sigma2 < 0.0, 0.0, sigma2)
    sigma = np.sqrt(sigma2)  # (N,)
    # draw normal noise per particle & per dimension
    xi = np.random.normal(size=(N,3))
    velocities = (c * velocities) + (sigma[:,None] * xi)
    

    
    ###IF PULL UNCOMMENT BELOW
    #velocities[-1] = np.array([v_pull_st, 0, 0]).reshape(1, 3)

    # Step A (half drift): x += 0.5*dt * v
    positions = positions + 0.5 * dt * velocities

    # Step B (half kick) with updated forces
    F = total_force_contribution(positions,
         idx_pull, k_trap, r_trap0, v_pull,
         angle_triple_indices, k_theta, theta_eq,
         angle_quadruple_indices, k_phi, phi_eq,
         k_r, r_eq, keys, hydroph_m, radius, residue_names)
    velocities = velocities + 0.5 * dt * (F * inv_m[:,None])


    ###IF PULL UNCOMMENT BELOW
    # velocities[-1] = np.array([v_pull_st, 0, 0]).reshape(1, 3)
    # positions[0] = initial_positions[0]

    return positions, velocities

v_pull = -1e-2
velocities = np.zeros_like(positions)

#contour length, pulling period, and velocity set up
contour_length = get_contour_length(positions)
print(contour_length)
pull_period = pulling_time(positions, v_pull, dt)
print(pull_period)

print(dt)
num_periods = 1
tot_duration = num_periods*pull_period
time_steps = tot_duration/dt
print('time steps')
print(time_steps)

settle_time = 5000
settle_timesteps = int(settle_time/dt)

for settle_t in range(settle_timesteps):
    F = total_force_contribution(positions,
          idx_pull, k_trap, r_trap0, v_pull,
          angle_triple_indices, k_theta, theta_eq,
          angle_quadruple_indices, k_phi, phi_eq,
          k_r, r_eq, keys, hydroph_m, radius, residue_names)
    
    positions, velocities = baoab_step(positions, velocities, masses_md, dt, gamma, T, F, v_pull, time, pull_period)
    
    if settle_t% 100 ==0:
        print(settle_t)

#Live visualization plot
# plt.ion()
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# line, = ax.plot([], [], [], 'k-', lw=2)  # backbone line
# points = ax.scatter([], [], [], c='r', s=40)  # Cα atoms

# Initial axis limits
# padding = 5.0
# ax.set_xlim(np.min(positions[:,0])-padding, np.max(positions[:,0])+padding)
# ax.set_ylim(np.min(positions[:,1])-padding, np.max(positions[:,1])+padding)
# ax.set_zlim(np.min(positions[:,2])-padding, np.max(positions[:,2])+padding)
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

extension_list = []
total_system_force_list = [[] for period in range(num_periods)]
total_system_force_x_list = [[] for period in range(num_periods)]
time_list = []

# F_pull_kjmol = 50
# F_pull_pico_newtons = F_pull_kjmol *1.66
# F_pull = F_pull_kjmol/N_A

time_steps = 75000

F_pull_list_kjmol = np.array([1, 2, 2.5, 3, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8])
F_pull_list_pN = F_pull_list_kjmol*1.66
F_pull_list = F_pull_list_kjmol/N_A

equilibrium = []

for F_index in range(len(F_pull_list)):
    F_pull = F_pull_list[F_index]
    
    for time_step in range(int(time_steps)):
      F = total_force_contribution(positions,
            idx_pull, k_trap, r_trap0, v_pull,
            angle_triple_indices, k_theta, theta_eq,
            angle_quadruple_indices, k_phi, phi_eq,
            k_r, r_eq, keys, hydroph_m, radius, residue_names)
      F[-1, 0] -= F_pull
      
      positions, velocities = baoab_step_pull(positions, velocities, masses_md, dt, gamma, T, F, F_pull, time, pull_period)
      planar_angles = planar_bond_angles(positions, angle_triple_indices, degrees=True)
      di_angles = dihedral_angles(positions, angle_quadruple_indices, degrees = True)
      time += dt
      # if int(time)%10 == 0:
          
      #     position_diff = initial_positions - positions
      #     position_diff_list.append(np.linalg.norm(position_diff))
          
          
          
      if int(time_step)%1000 == 0: 
          print(time_step)
          print('force:')
          print(F_pull*N_A)
          
      if time_step >= 70000:
          if time_step%100 == 0:
              extension = positions[-1,0]
              extension_list.append(extension)
              
             
    
      # if time_step % 100 == 0:  # update plot every 1000 steps
              
      #       # total_system_force = np.sum(np.linalg.norm(F, axis = 1))
      #       # total_system_force_list[i].append(total_system_force)
      #       # total_system_force_x = np.sum(F[:,0])
      #       # total_system_force_x_list[i].append(total_system_force_x)
            
            
      #       extension_list.append(np.abs(extension))
            
      #       time_list.append(time)
      #       # Update backbone line
      #       line.set_data(positions[:,0], positions[:,1])
      #       line.set_3d_properties(positions[:,2])
    
      #       # # Update points
      #       points._offsets3d = (positions[:,0], positions[:,1], positions[:,2])
    
            # # Only update axes
            # ax.set_xlim(np.min(positions[:,0])-padding, np.max(positions[:,0])+padding)
            # ax.set_ylim(np.min(positions[:,1])-padding, np.max(positions[:,1])+padding)
            # ax.set_zlim(np.min(positions[:,2])-padding, np.max(positions[:,2])+padding)
    
            # plt.draw()
            # plt.pause(0.001)
            # print('extension')
            # print(np.abs(extension))
            # if time_step > 300:
            #     print('diff')
            #     print(np.abs(extension) - np.abs(extension_list[-2]))
    extension_list = extension_list         
    extension_mean = np.mean(extension_list)
    equilibrium.append(extension_mean)
    

 # print(i)

#plt.ioff()

# plt.figure(figsize=(8, 5))
# plt.plot(time_list, extension_list)
# plt.xlabel("Time (ps)")
# plt.ylabel("Extension (nm)")
# plt.title(f"Extension vs Time for F = {F_pull_pico_newtons:.2f} pN")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plot_protein_3d_interactive(positions, keys=None)


np.savez(
    "force_extension_data.npz",
    equilibrium=np.array(equilibrium, dtype=object),
    force=np.array(F_pull_list_pN, dtype=object),
    final_positions=np.array(positions, dtype=object)
)





# fig, axs = plt.subplots(i+1, 1)
# for plt_index in range(i+1):
#     axs[plt_index].plot(extension_list[plt_index], total_system_force_list[plt_index], label='total force')
#     axs[plt_index].set_title(f'force vs extension {plt_index+1}')
#     axs[plt_index].plot(extension_list[plt_index], total_system_force_x_list[plt_index], label='x force')
    
# plt.legend(bbox_to_anchor= (0, -2))
# plt.tight_layout()
# plt.show()



# #print(F)
# plt.plot(time_list, position_diff_list)
# position_diff = initial_positions - positions
# print(np.mean(np.linalg.norm(position_diff, axis=0)))
