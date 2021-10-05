

import numpy as np 

arr = np.arange(0,11)
# print(arr)                  # [ 0  1  2  3  4  5  6  7  8  9 10]


# arr[0:5]
# arr[:5]
# arr[:5]


""" broadcast """
arr[:5] = 100 
# print(arr)                  # [100 100 100 100 100   5   6   7   8   9  10]



#############################################################################################

""" slicing in np.array will affect original array """

original_arr = np.arange(0,21)
# print(orginal_arr)          # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]

slice_of_orifinal = original_arr[:10]
# print(slice_of_orifinal)           # 0 1 2 3 4 5 6 7 8 9]

slice_of_orifinal[:] = 99
# print(slice_of_orifinal)            # [99 99 99 99 99 99 99 99 99 99]

# print(orginal_arr)                  # [99 99 99 99 99 99 99 99 99 99 10 11 12 13 14 15 16 17 18 19 20]


""" use copy() to avoid changing original value """
original_arr = np.arange(0,21)
arr_copy = original_arr.copy()

arr_copy[:] = 77
# print(arr_copy)                 # [77 77 77 77 77 77 77 77 77 77 77 77 77 77 77 77 77 77 77 77 77]

# print(original_arr)              # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]



#############################################################################################

""" indexing in 2D """
arr_2D = np.array([[5, 10, 5], [20, 25, 30], [35, 40, 45]])
# print(arr_2D)

# # [[ 5 10  5]
# #  [20 25 30]
# #  [35 40 45]]


# print(arr_2D.shape)                 # (3, 3)

print(arr_2D[0])                # [ 5 10  5]
print(arr_2D[:2])               # first 2 row 
# [[ 5 10  5]
#  [20 25 30]]


# [row][colm]
# these two are same meaning 
print(arr_2D[1][2])             # 30
print(arr_2D[1,2])              # 30

# colm - all row, colm starting index 2
print(arr_2D[:,2])              # [ 5 30 45]

print(arr_2D[:2, 1:])
# [[10  5]
#  [25 30]]


#############################################################################################
""" conditional selection """

arr= np.arange(0,11)
# print(arr)                  # [ 0  1  2  3  4  5  6  7  8  9 10]

print(arr > 5)              # [False False False False False False  True  True  True  True  True]


""" filter """
bool_arr = arr > 5
# print(bool_arr)             # [False False False False False False  True  True  True  True  True]

print(arr[bool_arr])        # [ 6  7  8  9 10]



