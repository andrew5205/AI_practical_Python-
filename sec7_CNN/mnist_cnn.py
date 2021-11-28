
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

#########################################################################################################################
################                            Creating and Training data            #######################################
#########################################################################################################################
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


model = Sequential()

# CONVOLUTIONAL LAYER - filter = pow of 2 
# https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(28, 28, 1), activation='relu'))

# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))

# https://keras.io/metrics/
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# we can add in additional metrics https://keras.io/metrics/




from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Train the model
model.fit(x_train, y_cat_train, epochs=10, validation_data=(x_test, y_cat_test), callbacks=[early_stop])

#  (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# 2021-11-27 19:35:54.381419: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
# Epoch 1/10
# 1875/1875 [==============================] - 12s 6ms/step - loss: 0.1344 - accuracy: 0.9613 - val_loss: 0.0514 - val_accuracy: 0.9827
# Epoch 2/10
# 1875/1875 [==============================] - 12s 7ms/step - loss: 0.0470 - accuracy: 0.9852 - val_loss: 0.0422 - val_accuracy: 0.9872
# Epoch 3/10
# 1875/1875 [==============================] - 13s 7ms/step - loss: 0.0303 - accuracy: 0.9909 - val_loss: 0.0386 - val_accuracy: 0.9865
# Epoch 4/10
# 1875/1875 [==============================] - 13s 7ms/step - loss: 0.0198 - accuracy: 0.9936 - val_loss: 0.0386 - val_accuracy: 0.9880
# Epoch 5/10
# 1875/1875 [==============================] - 13s 7ms/step - loss: 0.0144 - accuracy: 0.9952 - val_loss: 0.0400 - val_accuracy: 0.9893
# Epoch 6/10
# 1875/1875 [==============================] - 13s 7ms/step - loss: 0.0107 - accuracy: 0.9964 - val_loss: 0.0378 - val_accuracy: 0.9889
# Epoch 7/10
# 1875/1875 [==============================] - 13s 7ms/step - loss: 0.0091 - accuracy: 0.9972 - val_loss: 0.0405 - val_accuracy: 0.9885
# Epoch 8/10
# 1875/1875 [==============================] - 13s 7ms/step - loss: 0.0067 - accuracy: 0.9979 - val_loss: 0.0547 - val_accuracy: 0.9873







#########################################################################################################################
################                            Evaluate the model             ##############################################
#########################################################################################################################


# print(model.metrics_names)          # ['loss', 'accuracy']


losses = pd.DataFrame(model.history.history)
print(losses.head())
#        loss  accuracy  val_loss  val_accuracy
# 0  0.134416  0.961283  0.051389        0.9827
# 1  0.046999  0.985167  0.042220        0.9872
# 2  0.030254  0.990900  0.038646        0.9865
# 3  0.019847  0.993633  0.038632        0.9880
# 4  0.014387  0.995250  0.039987        0.9893


losses[['accuracy', 'val_accuracy']].plot()
# plt.show()


losses[['loss', 'val_loss']].plot()
plt.show()

print(model.metrics_names)                                  # ['loss', 'accuracy']
print(model.evaluate(x_test, y_cat_test, verbose=0))        # [0.054672058671712875, 0.9872999787330627]


from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict_classes(x_test)

print(y_cat_test.shape)         # (10000, 10)

print(y_cat_test[0])            # array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.], dtype=float32)

print(predictions[0])           # 7

print(y_test)                   # array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)




print(classification_report(y_test, predictions))
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99       980
#            1       1.00      0.99      0.99      1135
#            2       0.98      0.99      0.98      1032
#            3       0.97      0.99      0.98      1010
#            4       1.00      0.98      0.99       982
#            5       0.98      0.99      0.99       892
#            6       0.98      0.99      0.99       958
#            7       0.99      0.98      0.99      1028
#            8       0.99      0.98      0.98       974
#            9       0.98      0.98      0.98      1009

#     accuracy                           0.99     10000
#    macro avg       0.99      0.99      0.99     10000
# weighted avg       0.99      0.99      0.99     10000



print(confusion_matrix(y_test,predictions))
# [[ 975    0    1    0    0    0    2    1    1    0]
#  [   0 1122    1    4    0    1    4    1    2    0]
#  [   6    1 1013    1    1    0    0    5    5    0]
#  [   0    0    1 1003    0    2    0    0    4    0]
#  [   0    0    0    0  978    0    0    0    0    4]
#  [   2    0    1   16    0  869    3    0    1    0]
#  [   8    1    0    0    2    1  943    0    3    0]
#  [   0    0    6    1    0    0    0 1013    2    6]
#  [   2    0    1    0    1    0    0    0  965    5]
#  [   4    0    0    1    6    1    0    3    2  992]]


import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions), annot=True)
# https://github.com/matplotlib/matplotlib/issues/14751
# plt.show()




""" 
# Predicting a given image
"""
my_number = x_test[0]
plt.imshow(my_number.reshape(28,28))
plt.show()

# SHAPE --> (num_images, width, height, color_channels)
model.predict_classes(my_number.reshape(1, 28, 8, 1))             # array([7], dtype=int64)

