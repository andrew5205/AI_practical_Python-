import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print(x_train.shape)            # (50000, 32, 32, 3)
# print(x_train[0].shape)         # (32, 32, 3)


# FROG
plt.imshow(x_train[0])

# HORSE
plt.imshow(x_train[12])


""" pre processing """
# print(x_train[0])

# print(x_train[0].shape)         # (32, 32, 3)
# print(x_train.max())            # 255

x_train = x_train / 225
x_test = x_test / 255

# print(x_train.shape)            # (50000, 32, 32, 3)
# print(x_test.shape)             # (10000, 32, 32, 3)

# # y_test
""" labeling data """
from tensorflow.keras.utils import to_categorical

print(y_train.shape)            # (50000, 1)
print(y_train[0])               # array([6], dtype=uint8)

y_cat_train = to_categorical(y_train, 10)

print(y_cat_train.shape)            # (50000, 10)
print(y_cat_train[0])               # array([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.], dtype=float32)

y_cat_test = to_categorical(y_test,10)


""" Building the model"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

model = Sequential()

## FIRST SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

## SECOND SET OF LAYERS

# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4), input_shape=(32, 32, 3), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))

# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())

# 256 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(256, activation='relu'))

# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# print(model.summary())
#     # Model: "sequential"
#     # _________________________________________________________________
#     # Layer (type)                 Output Shape              Param #   
#     # =================================================================
#     # conv2d (Conv2D)              (None, 29, 29, 32)        1568      
#     # _________________________________________________________________
#     # max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
#     # _________________________________________________________________
#     # conv2d_1 (Conv2D)            (None, 11, 11, 32)        16416     
#     # _________________________________________________________________
#     # max_pooling2d_1 (MaxPooling2 (None, 5, 5, 32)          0         
#     # _________________________________________________________________
#     # flatten (Flatten)            (None, 800)               0         
#     # _________________________________________________________________
#     # dense (Dense)                (None, 256)               205056    
#     # _________________________________________________________________
#     # dense_1 (Dense)              (None, 10)                2570      
#     # =================================================================
#     # Total params: 225,610
#     # Trainable params: 225,610
#     # Non-trainable params: 0




from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit(x_train, y_cat_train, epochs=15, validation_data=(x_test, y_cat_test), callbacks=[early_stop]) 
        # 2021-12-03 20:01:08.534546: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
        # Epoch 1/15
        # 1563/1563 [==============================] - 29s 18ms/step - loss: 1.5070 - accuracy: 0.4613 - val_loss: 1.3649 - val_accuracy: 0.5128
        # Epoch 2/15
        # 1563/1563 [==============================] - 33s 21ms/step - loss: 1.1493 - accuracy: 0.5961 - val_loss: 1.1697 - val_accuracy: 0.5890
        # Epoch 3/15
        # 1563/1563 [==============================] - 32s 20ms/step - loss: 0.9950 - accuracy: 0.6547 - val_loss: 1.1772 - val_accuracy: 0.5921
        # Epoch 4/15
        # 1563/1563 [==============================] - 32s 20ms/step - loss: 0.8887 - accuracy: 0.6939 - val_loss: 1.0021 - val_accuracy: 0.6636
        # Epoch 5/15
        # 1563/1563 [==============================] - 30s 19ms/step - loss: 0.8108 - accuracy: 0.7212 - val_loss: 1.0736 - val_accuracy: 0.6366
        # Epoch 6/15
        # 1563/1563 [==============================] - 29s 19ms/step - loss: 0.7485 - accuracy: 0.7445 - val_loss: 1.0585 - val_accuracy: 0.6541
        # Epoch 7/15
        # 1563/1563 [==============================] - 29s 18ms/step - loss: 0.7019 - accuracy: 0.7629 - val_loss: 0.9944 - val_accuracy: 0.6784
        # Epoch 8/15
        # 1563/1563 [==============================] - 29s 19ms/step - loss: 0.6555 - accuracy: 0.7763 - val_loss: 1.0589 - val_accuracy: 0.6685
        # Epoch 9/15
        # 1563/1563 [==============================] - 29s 19ms/step - loss: 0.6234 - accuracy: 0.7895 - val_loss: 1.0615 - val_accuracy: 0.6698
        # Epoch 10/15
        # 1563/1563 [==============================] - 29s 19ms/step - loss: 0.5968 - accuracy: 0.7969 - val_loss: 1.1436 - val_accuracy: 0.6655





















