from Bio.PDB import PDBParser
import plotly.graph_objects as go
import numpy as np


parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "1NCT.pdb")

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
    residue = structure[0][chain_id][resseq] 
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




def plot_protein_3d_interactive(positions, keys=None): #Plots in browser

    line_trace = go.Scatter3d(
        x=positions[:,0],
        y=positions[:,1],
        z=positions[:,2],
        mode='lines',
        line=dict(color='black', width=4),
        name='Backbone'
    )

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

def bond_vectors(positions):
    return positions[1:, :] - positions[:-1, :]

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
    a = positions[angle_triplets[:,0]]
    b = positions[angle_triplets[:,1]]
    c = positions[angle_triplets[:,2]]

    u = a - b
    v = c - b 

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
    
    

def dihedral_angles(positions, angle_quadruples, degrees=True):

    a = positions[angle_quadruples[:,0]]
    b = positions[angle_quadruples[:,1]]
    c = positions[angle_quadruples[:,2]]
    d = positions[angle_quadruples[:,3]]

    #bond vectors
    b1 = b - a
    b2 = c - b
    b3 = d - c 

    #normals
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    #norms
    n1_norm = np.linalg.norm(n1, axis=1)
    n2_norm = np.linalg.norm(n2, axis=1)
    b2_norm = np.linalg.norm(b2, axis=1)

    #safety
    mask = (n1_norm < 1e-12) | (n2_norm < 1e-12) | (b2_norm < 1e-12)


    n1u = n1 / n1_norm[:,None]
    n2u = n2 / n2_norm[:,None]
    b2u = b2 / b2_norm[:,None]


    m = np.cross(n1u, b2u)


    x = np.sum(n1u * n2u, axis=1)
    y = np.sum(m * n2u, axis=1)

    phi = np.arctan2(y, x)           # signed dihedral

    if degrees:
        phi = np.degrees(phi)

    phi[mask] = np.nan

    return phi

#Below are functions for the various forces which arise from contact & non_conttact potentials
#in addition to the external constant-velocity pulling introduced to unfold our protein

#Intramolecular forces between our amino acids, modeled as spring like
#using hooke's law with experimentally determined 'spring constant' K_r
def F_bonds(positions, k_r, r_eq):
    N = len(positions)

    #bond vectors
    rij = positions[1:] - positions[:-1]

    # Bond lengths
    r = np.linalg.norm(rij, axis=1)

    #protect against zero-length bonds
    eps = 1e-12
    r_safe = np.maximum(r, eps)

    rhat = rij / r_safe[:,None]

    r_eq = np.asarray(r_eq)
    if r_eq.ndim == 0:
        r_eq = np.full_like(r, r_eq)

    fmag = k_r * (r - r_eq)

    forces = fmag[:,None] * rhat

    F = np.zeros((N,3))
    F[:N-1] +=  forces
    F[1:   ] += -forces

    return F



def F_bb_angles(positions, angle_triples, k_theta, theta_eq):
    
    i = angle_triples[:,0]
    j = angle_triples[:,1]
    k = angle_triples[:,2]

    #vectors from central atom j -> i and j -> k
    rij = positions[i] - positions[j]
    rkj = positions[k] - positions[j]

    rij_norm = np.linalg.norm(rij, axis=1)
    rkj_norm = np.linalg.norm(rkj, axis=1)


    e_ij = rij / rij_norm[:,None]
    e_kj = rkj / rkj_norm[:,None]


    cos_theta = np.sum(e_ij * e_kj, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)


    theta = np.arccos(cos_theta)

    dU_dtheta = k_theta * (theta - theta_eq)

    # sin(theta) with stability fix
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    sin_theta = np.where(sin_theta < 1e-12, 1e-12, sin_theta)

    #standard MD angle force formulas (OpenMM/GROMACS/CHARMM)
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


def F_dihedrals(positions, dihedral_quads, k_phi, phi_eq):

    i = dihedral_quads[:,0]
    j = dihedral_quads[:,1]
    k = dihedral_quads[:,2]
    l = dihedral_quads[:,3]


    b1 = positions[i] - positions[j]
    b2 = positions[k] - positions[j]
    b3 = positions[l] - positions[k]


    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1, axis=1)
    n2_norm = np.linalg.norm(n2, axis=1)
    b2_norm = np.linalg.norm(b2, axis=1)


    n1u = n1 / n1_norm[:,None]
    n2u = n2 / n2_norm[:,None]
    b2u = b2 / b2_norm[:,None]

    m = np.cross(n1u, b2u)
    x = np.sum(n1u * n2u, axis=1)
    y = np.sum(m * n2u, axis=1)
    phi = np.arctan2(y, x)

    dU_dphi = k_phi * (phi - phi_eq)

    inv_n1 = 1.0 / n1_norm
    inv_n2 = 1.0 / n2_norm
    inv_b2 = 1.0 / b2_norm

    t1 =  (n1 * inv_n1[:,None]) * inv_b2[:,None]
    t2 =  (n2 * inv_n2[:,None]) * inv_b2[:,None]

    Fi = -dU_dphi[:,None] * t1
    Fl =  dU_dphi[:,None] * t2
    Fj = -(Fi + np.cross(b2u, Fi) * b2_norm[:,None])
    Fk = -(Fl + np.cross(b2u, Fl) * b2_norm[:,None])

    F = np.zeros_like(positions)
    np.add.at(F, i, Fi)
    np.add.at(F, j, Fj)
    np.add.at(F, k, Fk)
    np.add.at(F, l, Fl)

    return F



#Non contact force contributions from Lennard-Jones & Coulombic (electrostatic) potentials
def non_contact_forces(positions, pairs, A, B, charges, epsilon):

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
        idx_pull, k_trap, r_trap0,
        planar_triples, k_theta, theta_eq,
        dihedral_quads, k_phi, phi_eq,
        k_r, r_eq, keys, hydroph_m, radius, residue_names):

    F  = F_bonds(positions, k_r, r_eq)
    F += F_bb_angles(positions, planar_triples, k_theta, theta_eq)
    F += F_dihedrals(positions, dihedral_quads, k_phi, phi_eq)
    F += nb_forces(positions, keys, residue_names, hydroph_m, radius, N_A, charge_map)

    return F

def per_molecule(k_kj_per_mol):
    return k_kj_per_mol / N_A

def mass_to_md_units(mass_amu):
    kg = mass_amu * amu_to_kg
    return kg * 1000.0 
 

def baoab_step_pull(positions, velocities, masses_md, dt, gamma, T, F, F_pull, time):

    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    N = positions.shape[0]

    masses_md = np.asarray(masses_md, dtype=float)
    if masses_md.ndim == 0:
        masses_md = np.full(N, masses_md)

    inv_m = 1.0 / masses_md 

    ##B##
    velocities = velocities + 0.5 * dt * (F * inv_m[:,None])
    

    ##A##
    positions = positions + 0.5 * dt * velocities

    ##O##
    c = np.exp(-gamma * dt)
    sigma2 = (1.0 - c*c) * (kB_per_particle * T) * inv_m 
    sigma2 = np.where(sigma2 < 0.0, 0.0, sigma2)
    sigma = np.sqrt(sigma2)  # (N,)
    xi = np.random.normal(size=(N,3))
    velocities = (c * velocities) + (sigma[:,None] * xi)


    ##A##
    positions = positions + 0.5 * dt * velocities

    ##B##
    F = total_force_contribution(positions,
         idx_pull, k_trap, r_trap0,
         angle_triple_indices, k_theta, theta_eq,
         angle_quadruple_indices, k_phi, phi_eq,
         k_r, r_eq, keys, hydroph_m, radius, residue_names)
    F[-1, 0] -= F_pull #Pulling force on last bead
    velocities = velocities + 0.5 * dt * (F * inv_m[:,None])
    
    # Anchored end 
    positions[0] = initial_positions[0]

    return positions, velocities

def baoab_step(positions, velocities, masses_md, dt, gamma, T, F, time):
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    N = positions.shape[0]

    masses_md = np.asarray(masses_md, dtype=float)
    if masses_md.ndim == 0:
        masses_md = np.full(N, masses_md)

    inv_m = 1.0 / masses_md

    ##B##
    
    velocities = velocities + 0.5 * dt * (F * inv_m[:,None])

    ##A##
    positions = positions + 0.5 * dt * velocities

    ##O##
    c = np.exp(-gamma * dt)
    sigma2 = (1.0 - c*c) * (kB_per_particle * T) * inv_m 
    sigma2 = np.where(sigma2 < 0.0, 0.0, sigma2)
    sigma = np.sqrt(sigma2)  # (N,)
    xi = np.random.normal(size=(N,3))
    velocities = (c * velocities) + (sigma[:,None] * xi)


    ##A##
    positions = positions + 0.5 * dt * velocities

    ##B##
    F = total_force_contribution(positions,
         idx_pull, k_trap, r_trap0,
         angle_triple_indices, k_theta, theta_eq,
         angle_quadruple_indices, k_phi, phi_eq,
         k_r, r_eq, keys, hydroph_m, radius, residue_names)
    velocities = velocities + 0.5 * dt * (F * inv_m[:,None])


    return positions, velocities


N = len(positions)

angle_triple_indices = angle_triplets(N)
angle_quadruple_indices = angle_quadruples(N)
positions = positions/10

#model parameters
radius = 0.2 # bead radii 
gamma = 5
gammas = np.full(N, gamma) 
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
time_list = []
position_diff_list = []

N_A = 6.02214076e23
kB_per_particle = 1.380649e-26
amu_to_kg = 1.66053906660e-27

k_r = per_molecule(500.0)
k_theta = per_molecule(30.0)
k_phi = per_molecule(2.0)

#Convert masses
mass_amu = 110.0
m_md = mass_to_md_units(mass_amu) 
masses_md = np.full(N, m_md)   

velocities = np.zeros_like(positions)


settle_time = 5000
settle_timesteps = int(settle_time/dt)

for settle_t in range(settle_timesteps):
    F = total_force_contribution(positions,
          idx_pull, k_trap, r_trap0,
          angle_triple_indices, k_theta, theta_eq,
          angle_quadruple_indices, k_phi, phi_eq,
          k_r, r_eq, keys, hydroph_m, radius, residue_names)
    
    positions, velocities = baoab_step(positions, velocities, masses_md, dt, gamma, T, F, time)
    
    if settle_t% 100 ==0:
        print(settle_t)


time_list = []

time_steps = 75000

F_pull_list_kjmol = np.array([1, 2, 2.5, 3, 3.5, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8])
F_pull_list_pN = F_pull_list_kjmol*1.66
F_pull_list = F_pull_list_kjmol/N_A

equilibrium = []

for F_index in range(len(F_pull_list)):
    F_pull = F_pull_list[F_index]
    extension_list = []
    
    for time_step in range(int(time_steps)):
      F = total_force_contribution(positions,
            idx_pull, k_trap, r_trap0,
            angle_triple_indices, k_theta, theta_eq,
            angle_quadruple_indices, k_phi, phi_eq,
            k_r, r_eq, keys, hydroph_m, radius, residue_names)
      F[-1, 0] -= F_pull
      
      positions, velocities = baoab_step_pull(positions, velocities, masses_md, dt, gamma, T, F, F_pull, time)
      planar_angles = planar_bond_angles(positions, angle_triple_indices, degrees=True)
      di_angles = dihedral_angles(positions, angle_quadruple_indices, degrees = True)
      time += dt
          
          
          
      if int(time_step)%1000 == 0: 
          print(time_step)
          print('force:')
          print(F_pull*N_A)
          
      if time_step >= 70000:
          if time_step%100 == 0:
              extension = positions[-1,0]
              extension_list.append(extension)
              
             
    extension_list = extension_list         
    extension_mean = np.mean(extension_list)
    equilibrium.append(extension_mean)
    


np.savez(
    "force_extension_data.npz",
    equilibrium=np.array(equilibrium, dtype=object),
    force=np.array(F_pull_list_pN, dtype=object),
    final_positions=np.array(positions, dtype=object)
)



