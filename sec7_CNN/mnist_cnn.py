
# setosa.io/ev/image-kernels/

# MNIST is well known CNN data set, hand written 0`9 `
# 60k training img
# 10k test img

# consider of the entire group of 60k img as 4-dimensional array, 28X28 pixels
# (samples, X, y, channel) -> (60000, 28, 28, 1)
# gray scale: channel == 1, 
# color scale: channel == 3

# color maps: 
# https://matplotlib.org/stable/tutorials/colors/colormaps.html

# colors:
# https://matplotlib.org/stable/tutorials/colors/colors.html



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


""" Read the data """
from tensorflow.keras.datasets import mnist

# load mnist data, and straight use tuple to seperate train/ test data set 
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train.shape
# print(x_train.shape)     # (60000, 28, 28)


single_image = x_train[0]
# print('single_image shape is:{}'.format(single_image.shape))            # single_image shape is:(28, 28)

# print(single_image)
# # [[  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   3  18  18  18 126 136
# #   175  26 166 255 247 127   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0  30  36  94 154 170 253 253 253 253 253
# #   225 172 253 242 195  64   0   0   0   0]
# #  [  0   0   0   0   0   0   0  49 238 253 253 253 253 253 253 253 253 251
# #    93  82  82  56  39   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0  18 219 253 253 253 253 253 198 182 247 241
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0  80 156 107 253 253 205  11   0  43 154
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0  14   1 154 253  90   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0 139 253 190   2   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0  11 190 253  70   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0  35 241 225 160 108   1
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0  81 240 253 253 119
# #    25   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  45 186 253 253
# #   150  27   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  16  93 252
# #   253 187   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0 249
# #   253 249  64   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0  46 130 183 253
# #   253 207   2   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0  39 148 229 253 253 253
# #   250 182   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0  24 114 221 253 253 253 253 201
# #    78   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0  23  66 213 253 253 253 253 198  81   2
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0  18 171 219 253 253 253 253 195  80   9   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0  55 172 226 253 253 253 253 244 133  11   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0 136 253 253 253 212 135 132  16   0   0   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]
# #  [  0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# #     0   0   0   0   0   0   0   0   0   0]]


plt.imshow(single_image)
# plt.show()


""" labeling """
# print(y_train)          # [5 0 4 ... 5 6 8]
# print(y_test)           # [7 2 1 ... 4 5 6]

""" 
Hmmm, looks like our labels are literally categories of numbers. 
We need to translate this to be "one hot encoded" so our CNN can understand, 
otherwise it will think this is some sort of regression problem on a continuous axis. 
Luckily , Keras has an easy to use function for this:
"""
from tensorflow.keras.utils import to_categorical

# print(y_train.shape)            # (60000,)

y_example = to_categorical(y_train)
# print(y_example.shape)          # (60000, 10)
# print(y_example[0])             # [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]


y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)


""" 
### Processing X Data
We should normalize the X data
"""
# print(single_image.max())           # 255
# print(single_image.min())           # 0

# to ratio
x_train = x_train/255
x_test = x_test/255

scaled_single = x_train[0]
# print(scaled_single.max())          # 1.0

plt.imshow(scaled_single)
# plt.show()



""" 
## Reshaping the Data

Right now our data is 60,000 images stored in 28 by 28 pixel array formation. 

This is correct for a CNN, but we need to add one more dimension to show we're dealing with 1 RGB channel 
(since technically the images are in black and white, only showing values from 0-255 on a single channel), 
an color image would have 3 dimensions.
"""
# print(x_train.shape)            # (60000, 28, 28)
# print(x_test.shape)             # (10000, 28, 28)


x_train = x_train.reshape(60000, 28, 28, 1)
# print(x_train.shape)            # (60000, 28, 28, 1)

x_test = x_test.reshape(10000,28,28,1)
# print(x_test.shape)             # (10000, 28, 28, 1)




