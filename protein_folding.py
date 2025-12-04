from Bio.PDB import PDBParser
import numpy as np

parser = PDBParser(QUIET=True)
structure = parser.get_structure("protein", "1NCT.pdb")

# store Cα atoms in a simple dict: (chain, resseq) → np.array([x,y,z])
ca_positions = {}

for model in structure:
    for chain in model:
        for residue in chain:
            if "CA" in residue:  # only alpha-carbon
                ca_positions[(chain.id, residue.id[1])] = residue["CA"].coord
print(ca_positions)