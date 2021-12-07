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



# Careful, don't overwrite our file!
# model.save('cifar_10epochs.h5')
losses = pd.DataFrame(model.history.history)
# print(losses.head())
#         #     loss  accuracy  val_loss  val_accuracy
#         # 0  1.507017   0.46126  1.364891        0.5128
#         # 1  1.149331   0.59610  1.169666        0.5890
#         # 2  0.994966   0.65474  1.177166        0.5921
#         # 3  0.888658   0.69388  1.002123        0.6636
#         # 4  0.810752   0.72122  1.073580        0.6366


losses[['accuracy', 'val_accuracy']].plot()

losses[['loss', 'val_loss']].plot()
plt.show()


# print(model.metrics_names)          # ['loss', 'accuracy']
# print(model.evaluate(x_test,y_cat_test,verbose=0))                    # [1.0987777906417846, 0.6544]



from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict_classes(x_test)
# print(classification_report(y_test, predictions))
# #                 precision    recall  f1-score   support

# #            0       0.68      0.67      0.67      1000
# #            1       0.81      0.77      0.79      1000
# #            2       0.48      0.61      0.54      1000
# #            3       0.54      0.39      0.46      1000
# #            4       0.70      0.55      0.61      1000
# #            5       0.55      0.61      0.58      1000
# #            6       0.74      0.76      0.75      1000
# #            7       0.88      0.55      0.68      1000
# #            8       0.58      0.92      0.71      1000
# #            9       0.77      0.71      0.74      1000

# #     accuracy                           0.65     10000
# #    macro avg       0.67      0.65      0.65     10000
# # weighted avg       0.67      0.65      0.65     10000



confusion_matrix(y_test, predictions)
# print(confusion_matrix(y_test, predictions))
#         # array([[665,  25,  50,   5,   8,   9,   9,   3, 206,  20],
#         # [ 22, 769,  13,   9,   1,   5,  10,   1, 105,  65],
#         # [ 83,   8, 613,  35,  45,  71,  60,   9,  59,  17],
#         # [ 34,  22, 138, 394,  48, 210,  62,  11,  58,  23],
#         # [ 41,   4, 145,  49, 550,  58,  67,  24,  54,   8],
#         # [ 20,   7,  99, 131,  37, 614,  39,  16,  25,  12],
#         # [ 15,  10,  74,  52,  30,  19, 757,   1,  30,  12],
#         # [ 32,   7, 104,  44,  69, 122,  12, 548,  23,  39],
#         # [ 27,  13,  14,   4,   3,   3,   1,   0, 924,  11],
#         # [ 39,  89,  17,   8,   0,   9,   3,   7, 118, 710]], dtype=int64)


import seaborn as sns
plt.figure(figsize=(10,6))
sns.heatmap(confusion_matrix(y_test,predictions), annot=True)
# https://github.com/matplotlib/matplotlib/issues/14751


""" # Predicting a given image """
my_image = x_test[16]
plt.imshow(my_image)

# print(y_test[16])


# SHAPE --> (num_images,width,height,color_channels)
model.predict_classes(my_image.reshape(1,32,32,3))


# 5 is DOG
# https://www.cs.toronto.edu/~kriz/cifar.html








