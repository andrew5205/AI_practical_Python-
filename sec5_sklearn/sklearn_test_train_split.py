

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



df = pd.read_csv("Advertising.csv")
# print(df)

# #         TV  radio  newspaper  sales
# # 0    230.1   37.8       69.2   22.1
# # 1     44.5   39.3       45.1   10.4
# # 2     17.2   45.9       69.3    9.3
# # 3    151.5   41.3       58.5   18.5
# # 4    180.8   10.8       58.4   12.9
# # ..     ...    ...        ...    ...
# # 195   38.2    3.7       13.8    7.6
# # 196   94.2    4.9        8.1    9.7
# # 197  177.0    9.3        6.4   12.8
# # 198  283.6   42.0       66.2   25.5
# # 199  232.1    8.6        8.7   13.4

# # [200 rows x 4 columns]


""" 
## Train | Test Split Procedure 

0. Clean and adjust data as necessary for X and y
1. Split Data in Train/Test for both X and y
2. Fit/Train Scaler on Training X Data
3. Scale X Test Data
4. Create Model
5. Fit/Train Model on X Train Data
6. Evaluate Model on X Test Data (by creating predictions and comparing to Y_test)
7. Adjust Parameters as Necessary and repeat steps 5 and 6
"""

# CREATE X and y
# x - feature 
# y - label
X = df.drop('sales',axis=1)
y = df['sales']


# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# SCALE DATA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


""" Create Model """
from sklearn.linear_model import Ridge

# Poor Alpha Choice on purpose!
model = Ridge(alpha=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)



""" Evaluation """
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test,y_pred)
# print(mean_squared_error(y_test,y_pred))            # 7.341775789034129


""" Evaluation #2"""
model_two = Ridge(alpha=1)
model_two.fit(X_train, y_train)
y_pred_two = model_two.predict(X_test)
mean_squared_error(y_test, y_pred_two)
# print(mean_squared_error(y_test,y_pred_two))        # 2.3190215794287514




