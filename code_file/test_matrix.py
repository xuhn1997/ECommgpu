from scipy.sparse import lil_matrix

"""
   测试稀疏矩阵的一些用法
"""

mat = lil_matrix((10, 10), dtype=float)

mat[9, 8] = 1
mat[8, 9] = 1

mat1 = lil_matrix((10, 10), dtype=float)
mat1[8, 9] = 10
mat1[0, 7] = 99

mat22 = mat + mat1
mat22[8, 5] = 90
mat22[8, 6] = 100
print(mat22)

print("--------")
# print(mat22.getrow(8))  # 返回一个1*n的稀疏矩阵行向量
mat_8 = mat22.getrow(8)
print(mat_8)
print("--------")
print(mat_8.indices)
# mat_8_no = mat_8.nonzero()

tuples = zip(mat_8.indices, mat_8.data)
# for i, j in tuples:
#     print(i)
#     print(j)
#     print("________________")
