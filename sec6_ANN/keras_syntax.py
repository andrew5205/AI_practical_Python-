
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


""" read from csv """
df = pd.read_csv('./DATA/fake_reg.csv')
# print(df.head())

# #         price     feature1     feature2
# # 0  461.527929   999.787558   999.766096
# # 1  548.130011   998.861615  1001.042403
# # 2  410.297162  1000.070267   998.844015
# # 3  540.382220   999.952251  1000.440940
# # 4  546.024553  1000.446011  1000.338531

""" show plot """
sns.pairplot(df)
# plt.show()

#########################################################################################
""" split train/ test data """
from sklearn.model_selection import train_test_split

# Convert Pandas to Numpy for Keras
# Features
X = df[['feature1','feature2']].values

# Label
y = df['price'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# print(X_train.shape)            # (700, 2)
# print(X_test.shape)             # (300, 2)
# print(y_test.shape)             # (300,)


#######################################################################################################################
""" scaling/ normalzing """
from sklearn.preprocessing import MinMaxScaler

# help(MinMaxScaler)
# # q to quit on MAC


scaler = MinMaxScaler()


# Notice to prevent data leakage from the test set, we only fit our scaler to the training set
scaler.fit(X_train)
# MinMaxScaler(copy=True, feature_range=(0, 1))

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


#######################################################################################################################
#######################################################################################################################
""" creating model - keras """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation

# help(Sequential)
# help(Dense)
# #  q to quit in on MAC


# """ regular densely connected NN layer - as a list  """
# model = Sequential([
#     Dense(units=2),
#     Dense(units=2),
#     Dense(units=2)
# ])



""" regular densely connected NN layer - add separately  
quick reaction on layers
"""
model = Sequential()

model.add(Dense(4, activation='relu'))                          # layer 1 
model.add(Dense(4, activation='relu'))                          # layer 2 
model.add(Dense(4, activation='relu'))                          # layer 3 

# Final output node for prediction
model.add(Dense(1))                                             # layer 4 -> final layer -> output
# print('num of layers:{}'.format(len(model.layers)))            # num of layers:4

model.compile(optimizer='rmsprop', loss='mse')

""" 
### Choosing an optimizer and loss
Keep in mind what kind of problem you are trying to solve:

    # For a multi-class classification problem
    model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    # For a binary classification problem
    model.compile(optimizer='rmsprop',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

    # For a mean squared error regression problem
    model.compile(optimizer='rmsprop',
                    loss='mse')
"""


model.fit(X_train,y_train,epochs=250)
""" 
# Training
Below are some common definitions that are necessary to know and understand to correctly utilize Keras:

* Sample: one element of a dataset.
    * Example: one image is a sample in a convolutional network
    * Example: one audio file is a sample for a speech recognition model
* Batch: a set of N samples. The samples in a batch are processed independently, in parallel. If training, a batch results in only one update to the model.A batch generally approximates the distribution of the input data better than a single input. The larger the batch, the better the approximation; however, it is also true that the batch will take longer to process and will still result in only one update. For inference (evaluate/predict), it is recommended to pick a batch size that is as large as you can afford without going out of memory (since larger batches will usually result in faster evaluation/prediction).
* Epoch: an arbitrary cutoff, generally defined as "one pass over the entire dataset", used to separate training into distinct phases, which is useful for logging and periodic evaluation.
* When using validation_data or validation_split with the fit method of Keras models, evaluation will be run at the end of every epoch.
* Within Keras, there is the ability to add callbacks specifically designed to be run at the end of an epoch. Examples of these are learning rate changes and model checkpointing (saving).
"""
# print(model.fit(X_train,y_train,epochs=250))

# # 2021-10-22 21:03:48.464858: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
# # To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
# # 2021-10-22 21:03:48.634632: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:176] None of the MLIR Optimization Passes are enabled (registered 2)
# # Epoch 1/250
# # 22/22 [==============================] - 1s 917us/step - loss: 256610.9688
# # Epoch 2/250
# # 22/22 [==============================] - 0s 865us/step - loss: 256512.5781
# # Epoch 3/250
# # 22/22 [==============================] - 0s 991us/step - loss: 256402.6719
# # ...
# # Epoch 248/250
# # 22/22 [==============================] - 0s 825us/step - loss: 24.0493
# # Epoch 249/250
# # 22/22 [==============================] - 0s 809us/step - loss: 24.5414
# # Epoch 250/250
# # 22/22 [==============================] - 0s 2ms/step - loss: 24.1580
# # <tensorflow.python.keras.callbacks.History object at 0x7fbf520864f0>



loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
plt.show()

# model.history.history
# # print(model.history.history)

# loss = model.history.history['loss']

# sns.lineplot(x=range(len(loss)),y=loss)
# plt.title("Training Loss per Epoch");

#######################################################################################################################
#######################################################################################################################
""" Model Evaluation """

training_score = model.evaluate(X_train, y_train, verbose=0)
# print(training_score)           # 24.31159019470215

test_score = model.evaluate(X_test,y_test,verbose=0)
# print(test_score)               # 26.416053771972656


test_predictions = model.predict(X_test)
# print(test_predictions)
test_predictions = pd.Series(test_predictions.reshape(300,))


pred_df = pd.DataFrame(y_test, columns=['Test Y'])
# print(pred_df)
# #         Test Y
# # 0    402.296319
# # 1    624.156198
# # 2    582.455066
# # 3    578.588606
# # 4    371.224104
# # ..          ...
# # 295  525.704657
# # 296  502.909473
# # 297  612.727910
# # 298  417.569725
# # 299  410.538250

# # [300 rows x 1 columns]


pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['Test Y','Model Predictions']
# print(pred_df)
# #         Test Y  Model Predictions
# # 0    402.296319         421.510712
# # 1    624.156198         608.720581
# # 2    582.455066         582.896301
# # 3    578.588606         562.156067
# # 4    371.224104         383.460938
# # ..          ...                ...
# # 295  525.704657         525.489075
# # 296  502.909473         507.554413
# # 297  612.727910         597.360779
# # 298  417.569725         433.809143
# # 299  410.538250         424.434814

# # [300 rows x 2 columns]


# plot by predictions 
sns.scatterplot(x='Test Y',y='Model Predictions',data=pred_df)
plt.show()

# pred_df['Error'] = pred_df['Test Y'] - pred_df['Model Predictions']
# sns.distplot(pred_df['Error'],bins=50)



""" MAE vs MSE """
from sklearn.metrics import mean_absolute_error, mean_squared_error
mean_absolute_error(pred_df['Test Y'],pred_df['Model Predictions'])
# print(mean_absolute_error(pred_df['Test Y'],pred_df['Model Predictions']))      # 4.107803803171395

mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions'])
# print(mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions']))       # 25.850322372690748
mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions']) ** 0.5
# print(mean_squared_error(pred_df['Test Y'],pred_df['Model Predictions']) ** 0.5)    # 5.042244275641468


# # Essentially the same thing, difference just due to precision
# test_score
# #RMSE
# test_score ** 0.5




#######################################################################################################################
#######################################################################################################################
""" Prediction on brand new data """
# [[Feature1, Feature2]]
new_gem = [[998,1000]]

# Don't forget to scale!
scaler.transform(new_gem)
new_gem = scaler.transform(new_gem)
# print(new_gem)            # [[0.14117652 0.53968792]]

model.predict(new_gem)
# print(model.predict(new_gem))           # [[420.43216]]



#######################################################################################################################
#######################################################################################################################
""" saving & loading models """
from tensorflow.keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

# load model .h5 
later_model = load_model('my_model.h5')

later_model.predict(new_gem)




