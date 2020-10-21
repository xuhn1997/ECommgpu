from scipy.sparse import *
import scipy as scp
import os

basedir = os.path.dirname(__file__)
print("basedir:" + basedir)
mat = None
for name in os.listdir('../tmpdata_iuf'):
    #  将存在于tmpdata下的所有组的稀疏矩阵进行合并
    print(name)
    if name[-3:] == 'npz':
        if mat is None:
            mat = load_npz('../tmpdata_iuf/' + name)
        else:
            mat += load_npz('../tmpdata_iuf/' + name)
# scp.sparse.save_npz('../commonMatrix/common_matrix.npz', mat)
scp.sparse.save_npz('../commonMatrix_iuf/common_matrix.npz', mat)

print("save successfully ......................")

