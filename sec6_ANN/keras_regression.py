import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./DATA/kc_house_data.csv')
# print(df.head)
# # <bound method NDFrame.head of                
# #                id        date     price  bedrooms  bathrooms  ...  zipcode      lat     long  sqft_living15  sqft_lot15
# # 0      7129300520  10/13/2014  221900.0         3       1.00  ...    98178  47.5112 -122.257           1340        5650
# # 1      6414100192   12/9/2014  538000.0         3       2.25  ...    98125  47.7210 -122.319           1690        7639
# # 2      5631500400   2/25/2015  180000.0         2       1.00  ...    98028  47.7379 -122.233           2720        8062
# # 3      2487200875   12/9/2014  604000.0         4       3.00  ...    98136  47.5208 -122.393           1360        5000
# # 4      1954400510   2/18/2015  510000.0         3       2.00  ...    98074  47.6168 -122.045           1800        7503
# # ...           ...         ...       ...       ...        ...  ...      ...      ...      ...            ...         ...
# # 21592   263000018   5/21/2014  360000.0         3       2.50  ...    98103  47.6993 -122.346           1530        1509
# # 21593  6600060120   2/23/2015  400000.0         4       2.50  ...    98146  47.5107 -122.362           1830        7200
# # 21594  1523300141   6/23/2014  402101.0         2       0.75  ...    98144  47.5944 -122.299           1020        2007
# # 21595   291310100   1/16/2015  400000.0         3       2.50  ...    98027  47.5345 -122.069           1410        1287
# # 21596  1523300157  10/15/2014  325000.0         2       0.75  ...    98144  47.5941 -122.299           1020        1357

# # [21597 rows x 21 columns]>

###################################################################################################################################
###################################################################################################################################

""" check misssing point"""
df.isnull().sum()
# print(df.isnull().sum())
# # id               0
# # date             0
# # price            0
# # bedrooms         0
# # bathrooms        0
# # sqft_living      0
# # sqft_lot         0
# # floors           0
# # waterfront       0
# # view             0
# # condition        0
# # grade            0
# # sqft_above       0
# # sqft_basement    0
# # yr_built         0
# # yr_renovated     0
# # zipcode          0
# # lat              0
# # long             0
# # sqft_living15    0
# # sqft_lot15       0
# # dtype: int64


df.describe().transpose()
# print(df.describe().transpose())
# #                 count          mean           std           min           25%           50%           75%           max
# # id             21597.0  4.580474e+09  2.876736e+09  1.000102e+06  2.123049e+09  3.904930e+09  7.308900e+09  9.900000e+09
# # price          21597.0  5.402966e+05  3.673681e+05  7.800000e+04  3.220000e+05  4.500000e+05  6.450000e+05  7.700000e+06
# # bedrooms       21597.0  3.373200e+00  9.262989e-01  1.000000e+00  3.000000e+00  3.000000e+00  4.000000e+00  3.300000e+01
# # bathrooms      21597.0  2.115826e+00  7.689843e-01  5.000000e-01  1.750000e+00  2.250000e+00  2.500000e+00  8.000000e+00
# # sqft_living    21597.0  2.080322e+03  9.181061e+02  3.700000e+02  1.430000e+03  1.910000e+03  2.550000e+03  1.354000e+04
# # sqft_lot       21597.0  1.509941e+04  4.141264e+04  5.200000e+02  5.040000e+03  7.618000e+03  1.068500e+04  1.651359e+06
# # floors         21597.0  1.494096e+00  5.396828e-01  1.000000e+00  1.000000e+00  1.500000e+00  2.000000e+00  3.500000e+00
# # waterfront     21597.0  7.547345e-03  8.654900e-02  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  1.000000e+00
# # view           21597.0  2.342918e-01  7.663898e-01  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  4.000000e+00
# # condition      21597.0  3.409825e+00  6.505456e-01  1.000000e+00  3.000000e+00  3.000000e+00  4.000000e+00  5.000000e+00
# # grade          21597.0  7.657915e+00  1.173200e+00  3.000000e+00  7.000000e+00  7.000000e+00  8.000000e+00  1.300000e+01
# # sqft_above     21597.0  1.788597e+03  8.277598e+02  3.700000e+02  1.190000e+03  1.560000e+03  2.210000e+03  9.410000e+03
# # sqft_basement  21597.0  2.917250e+02  4.426678e+02  0.000000e+00  0.000000e+00  0.000000e+00  5.600000e+02  4.820000e+03
# # yr_built       21597.0  1.971000e+03  2.937523e+01  1.900000e+03  1.951000e+03  1.975000e+03  1.997000e+03  2.015000e+03
# # yr_renovated   21597.0  8.446479e+01  4.018214e+02  0.000000e+00  0.000000e+00  0.000000e+00  0.000000e+00  2.015000e+03
# # zipcode        21597.0  9.807795e+04  5.351307e+01  9.800100e+04  9.803300e+04  9.806500e+04  9.811800e+04  9.819900e+04
# # lat            21597.0  4.756009e+01  1.385518e-01  4.715590e+01  4.747110e+01  4.757180e+01  4.767800e+01  4.777760e+01
# # long           21597.0 -1.222140e+02  1.407235e-01 -1.225190e+02 -1.223280e+02 -1.222310e+02 -1.221250e+02 -1.213150e+02
# # sqft_living15  21597.0  1.986620e+03  6.852305e+02  3.990000e+02  1.490000e+03  1.840000e+03  2.360000e+03  6.210000e+03
# # sqft_lot15     21597.0  1.275828e+04  2.727444e+04  6.510000e+02  5.100000e+03  7.620000e+03  1.008300e+04  8.712000e+05
###################################################################################################################################
###################################################################################################################################


plt.figure(figsize=(12,8))
sns.distplot(df['price'])
# plt.show()
# plt.close()

plt.figure(figsize=(12,8))
sns.countplot(df['bedrooms'])
# plt.show()
# plt.close()

df.corr()['price'].sort_values()
# print(df.corr()['price'].sort_values())
# # zipcode         -0.053402
# # id              -0.016772
# # long             0.022036
# # condition        0.036056
# # yr_built         0.053953
# # sqft_lot15       0.082845
# # sqft_lot         0.089876
# # yr_renovated     0.126424
# # floors           0.256804
# # waterfront       0.266398
# # lat              0.306692
# # bedrooms         0.308787
# # sqft_basement    0.323799
# # view             0.397370
# # bathrooms        0.525906
# # sqft_living15    0.585241
# # sqft_above       0.605368
# # grade            0.667951
# # sqft_living      0.701917
# # price            1.000000
# # Name: price, dtype: float64

plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='sqft_living', data=df)
# plt.show()
# plt.close()


plt.figure(figsize=(12,8))
sns.boxplot(x='bedrooms',y='price',data=df)
# plt.show()
# plt.close()


# print(df.columns)
# # Index(['id', 'date', 'price', 'bedrooms', 'bathrooms', 'sqft_living',
# #        'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
# #        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
# #        'lat', 'long', 'sqft_living15', 'sqft_lot15'],
# #       dtype='object')


plt.figure(figsize=(12,8))
sns.scatterplot(x='price', y='long', data=df)
# sns.scatterplot(x='price', y='lat', data=df)
# plt.show()
# plt.close()


""" (hue='') """
plt.figure(figsize=(12,8))
sns.scatterplot(x='long', y='lat', data=df, hue='price')
# plt.show()
# plt.close()


df.sort_values('price', ascending=False).head(20)
# print(len(df))                  # 21597
# print(len(df)*(0.01))           # 215.97


# reset data frame - grab 99% from the buttom  
non_top_1_perc = df.sort_values('price', ascending=False).iloc[216:]
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat', data=non_top_1_perc, hue='price', palette='RdYlGn', edgecolor=None, alpha=0.2)
# plt.show()
# plt.close()

sns.boxplot(x='waterfront',y='price',data=df)
# plt.show()
# plt.close()

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
""" feature engineering """

df = df.drop('id', axis=1)

""" common way to make datetime object """
# print(df['date'])
df['date'] = pd.to_datetime(df['date'])

df['year'] = df['date'].apply(lambda date:date.year)
# lambda function call => 
# # def year_extration(date):
# #     return date.year

df['month'] = df['date'].apply(lambda date:date.month)

# print(df.head())
# #        date     price  bedrooms  bathrooms  sqft_living  sqft_lot  ...      lat     long  sqft_living15  sqft_lot15  year  month
# # 0 2014-10-13  221900.0         3       1.00         1180      5650  ...  47.5112 -122.257           1340        5650  2014     10
# # 1 2014-12-09  538000.0         3       2.25         2570      7242  ...  47.7210 -122.319           1690        7639  2014     12
# # 2 2015-02-25  180000.0         2       1.00          770     10000  ...  47.7379 -122.233           2720        8062  2015      2
# # 3 2014-12-09  604000.0         4       3.00         1960      5000  ...  47.5208 -122.393           1360        5000  2014     12
# # 4 2015-02-18  510000.0         3       2.00         1680      8080  ...  47.6168 -122.045           1800        7503  2015      2

# # [5 rows x 22 columns]
###################################################################################################################################

plt.figure('price vs year', figsize=(10,6))
sns.boxplot(x='year', y='price', data=df)
# plt.show()


""" why this df.plot()"""
df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()
# plt.show()
df = df.drop('date',axis=1)

# df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)

#  could make sense due to scaling, higher should correlate to more value
df['yr_renovated'].value_counts()

df.head()


df['sqft_basement'].value_counts()
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

""" split test/ train"""
X = df.drop('price',axis=1)
y = df['price']

# # return as numpy, in case tf complains while validation_data 
# X = df.drop('price',axis=1).values
# # print(X)
# y = df['price'].values
# # print(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# fit + transfor in one step 
X_train= scaler.fit_transform(X_train)
# no need to fit in test set 
X_test = scaler.transform(X_test)

# print(X_train.shape)                    # (15117, 19)
# print(X_test.shape)                     # (6480, 19)



""" creat model """
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam

model = Sequential()

# based on X_train.shape, we have 19 incoming features, 
# nice to have 19 neurons for one layer as well 
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# validation_data() only for checking purposes, wont affect test and train data set 
model.fit(x=X_train, y=y_train.values, validation_data=(X_test, y_test.values), batch_size=128, epochs=400)

# Epoch 1/400
# 119/119 [==============================] - 1s 2ms/step - loss: 430233059328.0000 - val_loss: 418875146240.0000
# Epoch 2/400
# 119/119 [==============================] - 0s 1ms/step - loss: 428618416128.0000 - val_loss: 413279551488.0000
# ... 
# Epoch 399/400
# 119/119 [==============================] - 0s 1ms/step - loss: 28788975616.0000 - val_loss: 26365681664.0000
# Epoch 400/400
# 119/119 [==============================] - 0s 1ms/step - loss: 28769007616.0000 - val_loss: 26328829952.0000

losses = pd.DataFrame(model.history.history)
print(losses)
losses.plot()
# plt.show()

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

""" evaluation """

from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score



X_test
predictions = model.predict(X_test)

mean_absolute_error(y_test,predictions)                 # 101666.74811137635

np.sqrt(mean_squared_error(y_test,predictions))         # 162549.26352210157

explained_variance_score(y_test,predictions)            # 0.8009472831860807


# print(df['price'].mean())              # 540296.5735055795
# print(df['price'].median())            # 450000.0
# print(df['price'].describe())
# # count    2.159700e+04
# # mean     5.402966e+05
# # std      3.673681e+05
# # min      7.800000e+04
# # 25%      3.220000e+05
# # 50%      4.500000e+05
# # 75%      6.450000e+05
# # max      7.700000e+06
# # Name: price, dtype: float64

# Our predictions
plt.figure(figsize=(12,6))
plt.scatter(y_test, predictions)
# plt.show()

# Perfect predictions
plt.plot(y_test,y_test,'r')
# plt.show()


errors = y_test.values.reshape(6480, 1) - predictions
sns.distplot(errors)
# plt.show()



""" predicting on brand now data """
single_house = df.drop('price',axis=1).iloc[0]
# print(single_house)

single_house = scaler.transform(single_house.values.reshape(-1, 19))
# print(single_house)
# # [[0.2        0.08       0.08376422 0.00310751 0.         0.
# #   0.         0.5        0.4        0.10785619 0.         0.47826087
# #   0.         0.57149751 0.21760797 0.16193426 0.00582059 0.
# #   0.81818182]]

model.predict(single_house)
# print(model.predict(single_house))          # array([[283862.12]], dtype=float32)

df.iloc[0]
# print(df.iloc[0])
# # price            221900.0000
# # bedrooms              3.0000
# # bathrooms             1.0000
# # sqft_living        1180.0000
# # sqft_lot           5650.0000
# # floors                1.0000
# # waterfront            0.0000
# # view                  0.0000
# # condition             3.0000
# # grade                 7.0000
# # sqft_above         1180.0000
# # sqft_basement         0.0000
# # yr_built           1955.0000
# # yr_renovated          0.0000
# # lat                  47.5112
# # long               -122.2570
# # sqft_living15      1340.0000
# # sqft_lot15         5650.0000
# # year               2014.0000
# # month                10.0000
# # Name: 0, dtype: float64


