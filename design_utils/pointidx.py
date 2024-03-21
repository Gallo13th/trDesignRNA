import numpy as np

def rst_struc_trans(rsts1:str,rsts2:str):
    '''
    . : not matched
    '''
    ides = set(list(rsts1)).union(set(list(rsts2)))
    matched_ides = []
    rst1 = np.array(list(rsts1))
    rst2 = np.array(list(rsts2))
    for idx in ides:
        if idx == '.':
            continue
        if rsts1.count(idx) != rsts2.count(idx):
            raise ValueError('The number of matched residues in two restraints are different.')
        matched_idx1 = np.where(rst1 == idx)[0]
        matched_idx2 = np.where(rst2 == idx)[0]
        matched_pairs = np.array(list(zip(matched_idx1,matched_idx2)))
        matched_ides.append(matched_pairs)
    return matched_ides

def rst_ss_trans(rst:str):
    rst = np.array(list(rst))
    matched_idx = np.where(rst == '*')[0]
    return matched_idx

rst_seq_trans = rst_ss_trans

if __name__ == '__main__':
    rst1 = '111....1222'
    rst2 = '111.1222'
    print(rst_struc_trans(rst1,rst2))
    