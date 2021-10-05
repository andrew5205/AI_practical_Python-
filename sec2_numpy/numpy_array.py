
import numpy as np
from numpy.lib import twodim_base 


""" shape """

mylist = [1, 2, 3]
# print(type(mylist))             # <class 'list'>

myarr = np.array(mylist)
# print(type(myarr))            # <class 'numpy.ndarray'>


my_matrix = [[1,2,3], [4,5,6], [7,8,9]]

# apply to 2-D array
myMatrix = np.array(my_matrix)
# print(type(myMatrix))           # <class 'numpy.ndarray'>


# np.arange(start, end, step)
my_arange = np.arange(0, 10)
# print(my_arange)                # [0 1 2 3 4 5 6 7 8 9]



####################################################################################################
all_zeros = np.zeros(5)          # float
# print(all_zeros)               # [0. 0. 0. 0. 0.]
all_ones = np.ones(5)
# print(all_ones)                # [1. 1. 1. 1. 1.]



two_D_array = np.zeros((4,3))
# print(two_D_array)

# # [[0. 0. 0.]
# #  [0. 0. 0.]
# #  [0. 0. 0.]
# #  [0. 0. 0.]]



# evenly space
# np.linspace(start, end, # of point between, endpoint=True, retstep, dtpe, axis)
my_linspace = np.linspace(0, 10, 3)
# print(my_linspace)             # [ 0.  5. 10.]



####################################################################################################
""" square matrix """
my_sqr3_matrix = np.eye(3, k=0)
# print(my_sqr3_matrix)

# # [[1. 0. 0.]
# #  [0. 1. 0.]
# #  [0. 0. 1.]]

my_sqr3_matrix_k1 = np.eye(3, k=1)
# print(my_sqr3_matrix_k1)

# # [[0. 1. 0.]
# #  [0. 0. 1.]
# #  [0. 0. 0.]]


####################################################################################################
"""" random """
# uniform distribution between 0 and 1
rand = np.random.rand(1)
# print(rand)             # [0.64486256]


rand_2D = np.random.rand(3,2)
# print(rand_2D)

# # [[0.27547567 0.26509618]
# #  [0.91477918 0.95893509]
# #  [0.28744773 0.02348611]]


# std random distribution
std_randn = np.random.randn(10)
# print(std_randn)

# # [ 0.06537449  0.0911577  -0.58082101  0.25729495 -1.52956349  0.59968356
# #   1.65835772  1.05732454  0.1449304   0.24151478]


# in matrix
std_randn_matrix = np.random.randn(3,2)
# print(std_randn_matrix)

# # [[ 1.1756399   1.75200496]
# #  [-0.29805557 -1.34790747]
# #  [ 0.40680663  0.84434706]]


# randint(start, end, size)
randint = np.random.randint(0, 51, 5)
# print(randint)              # [22  0 18 50 30]

# randint(start, end, shape)
randint_2D = np.random.randint(0, 51, (3,2))
# print(randint_2D)

# # [[26 49]
# #  [ 0  4]
# #  [ 6 42]]


####################################################################################################
""" 
set seed - make a=same SET of dandom number 
seed(42) is arbitrary choice 
"""
np.random.seed(42)
set_seed_rand = np.random.rand(4)
# print(set_seed_rand)                # [0.37454012 0.95071431 0.73199394 0.59865848]


####################################################################################################
""" reshape """
arr = np.arange(0, 25)
# print(arr)
# # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
# #  24]
arr_new_shape = arr.reshape(5,5)
# print(arr_new_shape)

# # [[ 0  1  2  3  4]
# #  [ 5  6  7  8  9]
# #  [10 11 12 13 14]
# #  [15 16 17 18 19]
# #  [20 21 22 23 24]]



####################################################################################################
""" max, min, """

ranArr = np.random.randint(0, 101, 10)
# print(ranArr)               # [82 86 74 74 87 99 23  2 21 52]

print(ranArr.max())         # 99
print(ranArr.min())         # 2

""" argmax() argmin() - return index"""
print(ranArr.argmax())      # 5
print(ranArr.argmin())      # 7


print(ranArr.dtype)         # int64


""" shape, reshape() """
print(ranArr.shape)         # (10,)
new_sape = ranArr.reshape((5,2))
print(new_sape.shape)       # (5, 2)



