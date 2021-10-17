
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


#########################################################################################
""" scaling/ normalzing """
from sklearn.preprocessing import MinMaxScaler

# help(MinMaxScaler)
# # q to quit in MAC


scaler = MinMaxScaler()


# Notice to prevent data leakage from the test set, we only fit our scaler to the training set
scaler.fit(X_train)
# MinMaxScaler(copy=True, feature_range=(0, 1))

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)





