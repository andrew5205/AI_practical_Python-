

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
## Cross Validation with cross_val_score

split into sets
test: 70%
validation: 20%
test:10%


## Train | Validation | Test Split Procedure 

This is often also called a "hold-out" set, since you should not adjust parameters based on the final test set, 
but instead use it *only* for reporting final expected performance.

0. Clean and adjust data as necessary for X and y
1. Split Data in Train/Validation/Test for both X and y
2. Fit/Train Scaler on Training X Data
3. Scale X Eval Data
4. Create Model
5. Fit/Train Model on X Train Data
6. Evaluate Model on X Evaluation Data (by creating predictions and comparing to Y_eval)
7. Adjust Parameters as Necessary and repeat steps 5 and 6
8. Get final metrics on Test set (not allowed to go back and adjust after this!)
"""

## CREATE X and y
X = df.drop('sales',axis=1)
y = df['sales']

######################################################################
#### SPLIT TWICE! Here we create TRAIN | VALIDATION | TEST  #########
####################################################################

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split


# 70% of data is training data, set aside other 30%
X_train, X_OTHER, y_train, y_OTHER = train_test_split(X, y, test_size=0.3, random_state=101)

# test_size = 0.5 (50% of 30% of other ---> test = 15% of all data)
# Remaining 30% is split into evaluation and test sets
# Each is 15% of the original data size
X_eval, X_test, y_eval, y_test = train_test_split(X_OTHER, y_OTHER, test_size=0.5, random_state=101)

# print(len(df))              # 200 
# print(len(X_train))         # 140 
# print(len(X_train))         # 140
# print(len(X_eval))          # 30
# print(len(X_test))          # 30




# SCALE DATA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_eval = scaler.transform(X_eval)
X_test = scaler.transform(X_test)


# create model 
from sklearn.linear_model import Ridge

# Poor Alpha Choice on purpose!
model_one = Ridge(alpha=100)
model_one.fit(X_train, y_train)
y_eval_pred = model_one.predict(X_eval)


# check evaluation metric
from sklearn.metrics import mean_squared_error

mean_squared_error(y_eval,y_eval_pred)
# print(mean_squared_error(y_eval,y_eval_pred))           # 7.320101458823872


# Better Alpha Choice 
model_two = Ridge(alpha=1)
model_two.fit(X_train,y_train)
y_eval_pred_new = model_two.predict(X_eval)

mean_squared_error(y_eval, y_eval_pred)
# print(mean_squared_error(y_eval, y_eval_pred_new))          # 2.3837830750569866


""" final predict is made - no longer edit parameters after this point """  
y_final_test_pred = model_two.predict(X_test)
mean_squared_error(y_test, y_final_test_pred)
# print(mean_squared_error(y_test, y_final_test_pred))        # 2.254260083800517


