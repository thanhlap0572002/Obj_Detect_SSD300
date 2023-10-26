import numpy as np

X = np.array([4,3])

#Norm : chuẩn hóa để tính ra 1 đại lượng biểu diễn độ lớn của 1 vector

#L0Norm: số lượng các phần tử khác 0
#L0Norm = 2 ( đếm có bao nhiêu số trong ma trận)

L0norm = np.linalg.norm(X, ord=0)
print(L0norm)

#L1Norm: khoảng cách mahattan + các số trong ma trận

L1norm = np.linalg.norm(X, ord=1)
print(L1norm)

#L2Norm: khoảng cách  pytago

L2norm = np.linalg.norm(X, ord=2)
print(L2norm)


