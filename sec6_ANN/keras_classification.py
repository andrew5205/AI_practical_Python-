
# 1. classfication 
# 2. identify overfitting thru early stopping & dropout layers


import pandas as pd
import numpy as np


df = pd.read_csv('./DATA/cancer_classification.csv')

# print(df.info())
# # <class 'pandas.core.frame.DataFrame'>
# # RangeIndex: 569 entries, 0 to 568
# # Data columns (total 31 columns):
# #  #   Column                   Non-Null Count  Dtype  
# # ---  ------                   --------------  -----  
# #  0   mean radius              569 non-null    float64
# #  1   mean texture             569 non-null    float64
# #  2   mean perimeter           569 non-null    float64
# #  3   mean area                569 non-null    float64
# #  4   mean smoothness          569 non-null    float64
# #  5   mean compactness         569 non-null    float64
# #  6   mean concavity           569 non-null    float64
# #  7   mean concave points      569 non-null    float64
# #  8   mean symmetry            569 non-null    float64
# #  9   mean fractal dimension   569 non-null    float64
# #  10  radius error             569 non-null    float64
# #  11  texture error            569 non-null    float64
# #  12  perimeter error          569 non-null    float64
# #  13  area error               569 non-null    float64
# #  14  smoothness error         569 non-null    float64
# #  15  compactness error        569 non-null    float64
# #  16  concavity error          569 non-null    float64
# #  17  concave points error     569 non-null    float64
# #  18  symmetry error           569 non-null    float64
# #  19  fractal dimension error  569 non-null    float64
# #  20  worst radius             569 non-null    float64
# #  21  worst texture            569 non-null    float64
# #  22  worst perimeter          569 non-null    float64
# #  23  worst area               569 non-null    float64
# #  24  worst smoothness         569 non-null    float64
# #  25  worst compactness        569 non-null    float64
# #  26  worst concavity          569 non-null    float64
# #  27  worst concave points     569 non-null    float64
# #  28  worst symmetry           569 non-null    float64
# #  29  worst fractal dimension  569 non-null    float64
# #  30  benign_0__mal_1          569 non-null    int64  
# # dtypes: float64(30), int64(1)
# # memory usage: 137.9 KB

# # pd.transpose() - Transpose index and columns
# # index become features
# print(df.describe().transpose())
# #                          count        mean         std         min         25%         50%          75%         max
# # mean radius              569.0   14.127292    3.524049    6.981000   11.700000   13.370000    15.780000    28.11000
# # mean texture             569.0   19.289649    4.301036    9.710000   16.170000   18.840000    21.800000    39.28000
# # mean perimeter           569.0   91.969033   24.298981   43.790000   75.170000   86.240000   104.100000   188.50000
# # mean area                569.0  654.889104  351.914129  143.500000  420.300000  551.100000   782.700000  2501.00000
# # mean smoothness          569.0    0.096360    0.014064    0.052630    0.086370    0.095870     0.105300     0.16340
# # mean compactness         569.0    0.104341    0.052813    0.019380    0.064920    0.092630     0.130400     0.34540
# # mean concavity           569.0    0.088799    0.079720    0.000000    0.029560    0.061540     0.130700     0.42680
# # mean concave points      569.0    0.048919    0.038803    0.000000    0.020310    0.033500     0.074000     0.20120
# # mean symmetry            569.0    0.181162    0.027414    0.106000    0.161900    0.179200     0.195700     0.30400
# # mean fractal dimension   569.0    0.062798    0.007060    0.049960    0.057700    0.061540     0.066120     0.09744
# # radius error             569.0    0.405172    0.277313    0.111500    0.232400    0.324200     0.478900     2.87300
# # texture error            569.0    1.216853    0.551648    0.360200    0.833900    1.108000     1.474000     4.88500
# # perimeter error          569.0    2.866059    2.021855    0.757000    1.606000    2.287000     3.357000    21.98000
# # area error               569.0   40.337079   45.491006    6.802000   17.850000   24.530000    45.190000   542.20000
# # smoothness error         569.0    0.007041    0.003003    0.001713    0.005169    0.006380     0.008146     0.03113
# # compactness error        569.0    0.025478    0.017908    0.002252    0.013080    0.020450     0.032450     0.13540
# # concavity error          569.0    0.031894    0.030186    0.000000    0.015090    0.025890     0.042050     0.39600
# # concave points error     569.0    0.011796    0.006170    0.000000    0.007638    0.010930     0.014710     0.05279
# # symmetry error           569.0    0.020542    0.008266    0.007882    0.015160    0.018730     0.023480     0.07895
# # fractal dimension error  569.0    0.003795    0.002646    0.000895    0.002248    0.003187     0.004558     0.02984
# # worst radius             569.0   16.269190    4.833242    7.930000   13.010000   14.970000    18.790000    36.04000
# # worst texture            569.0   25.677223    6.146258   12.020000   21.080000   25.410000    29.720000    49.54000
# # worst perimeter          569.0  107.261213   33.602542   50.410000   84.110000   97.660000   125.400000   251.20000
# # worst area               569.0  880.583128  569.356993  185.200000  515.300000  686.500000  1084.000000  4254.00000
# # worst smoothness         569.0    0.132369    0.022832    0.071170    0.116600    0.131300     0.146000     0.22260
# # worst compactness        569.0    0.254265    0.157336    0.027290    0.147200    0.211900     0.339100     1.05800
# # worst concavity          569.0    0.272188    0.208624    0.000000    0.114500    0.226700     0.382900     1.25200
# # worst concave points     569.0    0.114606    0.065732    0.000000    0.064930    0.099930     0.161400     0.29100
# # worst symmetry           569.0    0.290076    0.061867    0.156500    0.250400    0.282200     0.317900     0.66380
# # worst fractal dimension  569.0    0.083946    0.018061    0.055040    0.071460    0.080040     0.092080     0.20750
# # benign_0__mal_1          569.0    0.627417    0.483918    0.000000    0.000000    1.000000     1.000000     1.00000



""" EDA """
import seaborn as sns
import matplotlib.pyplot as plt

# plt.figure(figsize=(12,6))
sns.countplot(x='benign_0__mal_1', data=df)
sns.heatmap(df.corr())
# plt.show()

df.corr()['benign_0__mal_1'].sort_values()
# print(df.corr()['benign_0__mal_1'].sort_values())
# # worst concave points      -0.793566
# # worst perimeter           -0.782914
# # mean concave points       -0.776614
# # worst radius              -0.776454
# # mean perimeter            -0.742636
# # worst area                -0.733825
# # mean radius               -0.730029
# # mean area                 -0.708984
# # mean concavity            -0.696360
# # worst concavity           -0.659610
# # mean compactness          -0.596534
# # worst compactness         -0.590998
# # radius error              -0.567134
# # perimeter error           -0.556141
# # area error                -0.548236
# # worst texture             -0.456903
# # worst smoothness          -0.421465
# # worst symmetry            -0.416294
# # mean texture              -0.415185
# # concave points error      -0.408042
# # mean smoothness           -0.358560
# # mean symmetry             -0.330499
# # worst fractal dimension   -0.323872
# # compactness error         -0.292999
# # concavity error           -0.253730
# # fractal dimension error   -0.077972
# # symmetry error             0.006522
# # texture error              0.008303
# # mean fractal dimension     0.012838
# # smoothness error           0.067016
# # benign_0__mal_1            1.000000
# # Name: benign_0__mal_1, dtype: float64


# plt.figure(figsize=(12,6))
df.corr()['benign_0__mal_1'].sort_values().plot(kind='bar')
# plt.show()


# plt.figure(figsize=(12,6))
# # drop the very last one, and compare corr
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
# plt.show()


####################################################################################################################
####################################################################################################################
####################################################################################################################

""" train and split """
X = df.drop('benign_0__mal_1', axis=1).values
y = df['benign_0__mal_1'].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



""" create model """
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

# print(X_train.shape)            # (426, 30)

model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=15, activation='relu'))

# one final layer 
# binary classification
model.add(Dense(units=1, activation='sigmoid'))

# For a binary classification problem
model.compile(loss='binary_crossentropy', optimizer='adam')



""" train the model """
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), verbose=1)


# model.history.history
model_loss = pd.DataFrame(model.history.history)
# print(model_loss)
# #          loss  val_loss
# # 0    0.686313  0.673294
# # 1    0.657916  0.644049
# # 2    0.625918  0.606659
# # 3    0.582010  0.552656
# # 4    0.525808  0.492546
# # ..        ...       ...
# # 595  0.001954  0.322886
# # 596  0.001822  0.325524
# # 597  0.001733  0.320392
# # 598  0.001657  0.341554
# # 599  0.001747  0.336945

# # [600 rows x 2 columns]

model_loss.plot()
# plt.show()


print('#########################################################################################')
print('#########################################################################################')
print(' ')
print('early stopping below')
print(' ')
print('#########################################################################################')
print('#########################################################################################')
####################################################################################################################
####################################################################################################################
####################################################################################################################
""" Early stopping - 
    automatically stop training based on a loss condition on the validation data pass thru model.fit() 
    
tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

"""

from tensorflow.keras.callbacks import EarlyStopping

# callback
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model = Sequential()
model.add(Dense(units=30, activation='relu'))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stop])


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
plt.show()


####################################################################################################################
####################################################################################################################
####################################################################################################################
""" adding drop out layers -
    added to layers to " turn off neuron during trsaining to prevent overfitting 
    each dropout layer will drop a user-defined percentage of neuruon units in the previous layer every batch 
"""

from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Dense(units=30, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=15, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stop])


model_loss = pd.DataFrame(model.history.history)
model_loss.plot()
# plt.show()


####################################################################################################################
####################################################################################################################
####################################################################################################################
""" model evaluaiton"""

from sklearn.metrics import classification_report, confusion_matrix

predictions = model.predict_classes(X_test)

# # https://en.wikipedia.org/wiki/Precision_and_recall
# print(classification_report(y_test, predictions))
# #               precision    recall  f1-score   support

# #            0       0.95      0.98      0.96        55
# #            1       0.99      0.97      0.98        88

# #     accuracy                           0.97       143
# #    macro avg       0.97      0.97      0.97       143
# # weighted avg       0.97      0.97      0.97       143



# print(confusion_matrix(y_test, predictions))
# # [[54  1]
# #  [ 3 85]]




