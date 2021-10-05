"""
Universal function - ufunc for ndarrays
https://numpy.org/doc/stable/reference/ufuncs.html
"""


import numpy as np 


""" element to element basis """

arr = np.arange(0, 10)
print(arr)              # [0 1 2 3 4 5 6 7 8 9]

print(arr + 5)          # [ 5  6  7  8  9 10 11 12 13 14]
print(arr * 5)          # [ 0  5 10 15 20 25 30 35 40 45]

# attention to shape 
print(arr+arr)          # [ 0  2  4  6  8 10 12 14 16 18]


print(np.sqrt(arr))
# [0.         1.         1.41421356 1.73205081 2.         2.23606798
#  2.44948974 2.64575131 2.82842712 3.        ]


""" 
numpy_op2.py:23: RuntimeWarning: divide by zero encountered in log
print(np.log(arr))
"""
print(np.log(arr))
# [      -inf 0.         0.69314718 1.09861229 1.38629436 1.60943791
#  1.79175947 1.94591015 2.07944154 2.19722458]



print(arr.sum())            # 45

print(arr.mean())           # 4.5

print(arr.max())            # 9

# variance
print(arr.var())            # 8.25

# standard deviation 
print(arr.std())            # 2.8722813232690143




#############################################################################################
""" quickest way to make 2D array => arange() + reshape() """
arr_2D = np.arange(0,25).reshape(5,5)
# print(arr_2D)

# # [[ 0  1  2  3  4]
# #  [ 5  6  7  8  9]
# #  [10 11 12 13 14]
# #  [15 16 17 18 19]
# #  [20 21 22 23 24]]



print(arr_2D.sum())             # 300


print(arr_2D.shape)                 # (5, 5)
# np.sum(axis=, dtype=, out=, keepdims=)
# axis=0 means row, based on shape
print(arr_2D.sum(axis=0))           # [50 55 60 65 70]
# axis=1 means colm, based on shape
print(arr_2D.sum(axis=1))           # [ 10  35  60  85 110]

# print(arr_2D.sum(axis=2))           #     return umr_sum(a, axis, dtype, out, keepdims, initial, where)
#                                     #     numpy.AxisError: axis 2 is out of bounds for array of dimension 2


