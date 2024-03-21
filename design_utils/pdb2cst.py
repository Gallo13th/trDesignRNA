import Bio.PDB
import numpy as np
from collections import defaultdict
import torch
bins_size = 1
dist_max = 40
dist_min = 3
dist_nbins = int((dist_max - dist_min) / bins_size + 1)
dist_bins = np.linspace(dist_min, dist_max, int((dist_max - dist_min) / bins_size + 1))
angle_bins = np.linspace(0.0, np.pi, 13)
dihe_bins = np.linspace(-np.pi, np.pi, 25)
angle1d_bins = np.linspace(70, 160, int((160 - 70) / 7.5 + 1)) * np.pi / 180

n_bins = {
    'C1\'': dist_nbins,
    'C4': dist_nbins,
    'P': dist_nbins,
    'N1': dist_nbins,
    'C3\'': dist_nbins,
}
bins = {
    'C1\'': dist_bins,
    'C4': dist_bins,
    'P': dist_bins,
    'N1': dist_bins,
    'C3\'': dist_bins,
}

def read_pdb_file(pdb_file):
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure('struct', pdb_file)
    return structure

def cal_restraint_map(structure):
    
    restraint_map = {
        'distance': {k: None for k in ['P', 'C3\'', 'C1\'', 'C4', 'N1']},       
    }
    # reindex the id to index
    ids = []
    for residue in structure.get_residues():
        ids.append(residue.id[1])
    ids = dict(zip(ids, range(len(ids))))
    for atom in ['P', 'C3\'', 'C1\'', 'C4', 'N1']:
        n_res = 0
        for residue in structure.get_residues():
            n_res += 1
        distance = np.zeros((n_res,n_res))
        coord = np.zeros((n_res, 3))
        for residue in structure.get_residues():
            try:
                coord[ids[residue.id[1]]] = residue[atom].get_coord()
            except:
                coord[ids[residue.id[1]]] = [np.nan, np.nan, np.nan]
        distance = np.linalg.norm(coord[:, None] - coord, axis=-1) 
        restraint_map['distance'][atom] = distance
    return restraint_map

def parse_labels(raw,device='cuda:0'):
    labels = defaultdict(dict)
    for a in ['P', 'C3\'', 'C1\'', 'C4', 'N1']:
        label = raw['distance'][a]
        binned = np.digitize(label, bins[a])
        binned[(binned >= n_bins[a]) & (~np.isnan(label))] = 0
        onehot = (np.arange(n_bins[a]) == binned[..., None]).astype(np.uint8)
        # the nan value treated as all zero
        onehot[np.isnan(label)] = 0
        labels['distance'][a] = torch.from_numpy(onehot).float().to(device)

    return labels

def pdb2cst(pdb_file):
    structure = read_pdb_file(pdb_file)
    rmap = cal_restraint_map(structure)
    labels = parse_labels(rmap)
    return labels

if __name__ == '__main__':
    pdb_file = "//wsl.localhost/Ubuntu-22.04/home/letgao/RNAdesign/trRosettaRNA_v1.1/design_utils/template.pdb"
    structure = read_pdb_file(pdb_file)
    rmap = cal_restraint_map(structure)
    labels = parse_labels(rmap)
    print(labels['distance'].keys())