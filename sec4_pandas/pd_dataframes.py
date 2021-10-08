
import numpy as np 
import pandas as pd 


# Make sure the seed is in the same cell as the random call
# https://stackoverflow.com/questions/21494489/what-does-numpy-random-seed0-do
np.random.seed(101)
mydata = np.random.randint(0,101,(4,3))
# print(mydata)

# # [[95 11 81]
# #  [70 63 87]
# #  [75  9 77]
# #  [40  4 63]]


myindex = ['CA','NY','AZ','TX']
mycolumns = ['Jan','Feb','Mar']

df = pd.DataFrame(data=mydata)
print(df)

#     0   1   2
# 0  95  11  81
# 1  70  63  87
# 2  75   9  77
# 3  40   4  63


df = pd.DataFrame(data=mydata,index=myindex)
print(df)

#      0   1   2
# CA  95  11  81
# NY  70  63  87
# AZ  75   9  77
# TX  40   4  63


df = pd.DataFrame(data=mydata,index=myindex,columns=mycolumns)
print(df)

#     Jan  Feb  Mar
# CA   95   11   81
# NY   70   63   87
# AZ   75    9   77
# TX   40    4   63


print(df.info())

# class 'pandas.core.frame.DataFrame'>
# Index: 4 entries, CA to TX
# Data columns (total 3 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   Jan     4 non-null      int64
#  1   Feb     4 non-null      int64
#  2   Mar     4 non-null      int64
# dtypes: int64(3)
# memory usage: 128.0+ bytes
# None


###############################################################################################################################


df = pd.read_csv('tips.csv')
print(df)

#      total_bill   tip     sex smoker   day    time  size  price_per_person          Payer Name         CC Number Payment ID
# 0         16.99  1.01  Female     No   Sun  Dinner     2              8.49  Christy Cunningham  3560325168603410    Sun2959
# 1         10.34  1.66    Male     No   Sun  Dinner     3              3.45      Douglas Tucker  4478071379779230    Sun4608
# 2         21.01  3.50    Male     No   Sun  Dinner     3              7.00      Travis Walters  6011812112971322    Sun4458
# 3         23.68  3.31    Male     No   Sun  Dinner     2             11.84    Nathaniel Harris  4676137647685994    Sun5260
# 4         24.59  3.61  Female     No   Sun  Dinner     4              6.15        Tonya Carter  4832732618637221    Sun2251
# ..          ...   ...     ...    ...   ...     ...   ...               ...                 ...               ...        ...
# 239       29.03  5.92    Male     No   Sat  Dinner     3              9.68       Michael Avila  5296068606052842    Sat2657
# 240       27.18  2.00  Female    Yes   Sat  Dinner     2             13.59      Monica Sanders  3506806155565404    Sat1766
# 241       22.67  2.00    Male    Yes   Sat  Dinner     2             11.34          Keith Wong  6011891618747196    Sat3880
# 242       17.82  1.75    Male     No   Sat  Dinner     2              8.91        Dennis Dixon     4375220550950      Sat17
# 243       18.78  3.00  Female     No  Thur  Dinner     2              9.39     Michelle Hardin  3511451626698139    Thur672

# [244 rows x 11 columns]


###############################################################################################################################
""" 
# print(df.columns)
# print(df.index)
# print(df.head(5))
# print(df.tail(3))
# print(df.info())
# print(df.describe())
# print(df.describe().transpose())
"""

print(df.columns)
# Index(['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size',
#        'price_per_person', 'Payer Name', 'CC Number', 'Payment ID'],
#       dtype='object')


print(df.index)
# RangeIndex(start=0, stop=244, step=1)


print(df.head())

#    total_bill   tip     sex smoker  day    time  size  price_per_person          Payer Name         CC Number Payment ID
# 0       16.99  1.01  Female     No  Sun  Dinner     2              8.49  Christy Cunningham  3560325168603410    Sun2959
# 1       10.34  1.66    Male     No  Sun  Dinner     3              3.45      Douglas Tucker  4478071379779230    Sun4608
# 2       21.01  3.50    Male     No  Sun  Dinner     3              7.00      Travis Walters  6011812112971322    Sun4458
# 3       23.68  3.31    Male     No  Sun  Dinner     2             11.84    Nathaniel Harris  4676137647685994    Sun5260
# 4       24.59  3.61  Female     No  Sun  Dinner     4              6.15        Tonya Carter  4832732618637221    Sun2251


print(df.tail(3))

#      total_bill   tip     sex smoker   day    time  size  price_per_person       Payer Name         CC Number Payment ID
# 241       22.67  2.00    Male    Yes   Sat  Dinner     2             11.34       Keith Wong  6011891618747196    Sat3880
# 242       17.82  1.75    Male     No   Sat  Dinner     2              8.91     Dennis Dixon     4375220550950      Sat17
# 243       18.78  3.00  Female     No  Thur  Dinner     2              9.39  Michelle Hardin  3511451626698139    Thur672


print(df.info())

# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 244 entries, 0 to 243
# Data columns (total 11 columns):
#  #   Column            Non-Null Count  Dtype  
# ---  ------            --------------  -----  
#  0   total_bill        244 non-null    float64
#  1   tip               244 non-null    float64
#  2   sex               244 non-null    object 
#  3   smoker            244 non-null    object 
#  4   day               244 non-null    object 
#  5   time              244 non-null    object 
#  6   size              244 non-null    int64  
#  7   price_per_person  244 non-null    float64
#  8   Payer Name        244 non-null    object 
#  9   CC Number         244 non-null    int64  
#  10  Payment ID        244 non-null    object 
# dtypes: float64(3), int64(2), object(6)
# memory usage: 21.1+ KB
# None


print(df.describe())

# (base) andrews-mbp-4:sec4_pandas andrew$ python pd_dataframe.py 
#        total_bill         tip        size  price_per_person     CC Number
# count  244.000000  244.000000  244.000000        244.000000  2.440000e+02
# mean    19.785943    2.998279    2.569672          7.888197  2.563496e+15
# std      8.902412    1.383638    0.951100          2.914234  2.369340e+15
# min      3.070000    1.000000    1.000000          2.880000  6.040679e+10
# 25%     13.347500    2.000000    2.000000          5.800000  3.040731e+13
# 50%     17.795000    2.900000    2.000000          7.255000  3.525318e+15
# 75%     24.127500    3.562500    3.000000          9.390000  4.553675e+15
# max     50.810000   10.000000    6.000000         20.270000  6.596454e+15



print(df.describe().transpose())

#                  count          mean           std           min           25%           50%           75%           max
# total_bill        244.0  1.978594e+01  8.902412e+00  3.070000e+00  1.334750e+01  1.779500e+01  2.412750e+01  5.081000e+01
# tip               244.0  2.998279e+00  1.383638e+00  1.000000e+00  2.000000e+00  2.900000e+00  3.562500e+00  1.000000e+01
# size              244.0  2.569672e+00  9.510998e-01  1.000000e+00  2.000000e+00  2.000000e+00  3.000000e+00  6.000000e+00
# price_per_person  244.0  7.888197e+00  2.914234e+00  2.880000e+00  5.800000e+00  7.255000e+00  9.390000e+00  2.027000e+01
# CC Number         244.0  2.563496e+15  2.369340e+15  6.040679e+10  3.040731e+13  3.525318e+15  4.553675e+15  6.596454e+15


###############################################################################################################################
###############################################################################################################################
""" 
# print(df['total_bill'])
# print(df[[col1, col2, col3]])

"""


print(df['total_bill'])

# 0      16.99
# 1      10.34
# 2      21.01
# 3      23.68
# 4      24.59
#        ...  
# 239    29.03
# 240    27.18
# 241    22.67
# 242    17.82
# 243    18.78
# Name: total_bill, Length: 244, dtype: float64


print(type(df['total_bill']))
# <class 'pandas.core.series.Series'>


# Note how its a python list of column names! Thus the double brackets.
print(df[['total_bill','tip']])

#     total_bill   tip
# 0         16.99  1.01
# 1         10.34  1.66
# 2         21.01  3.50
# 3         23.68  3.31
# 4         24.59  3.61
# ..          ...   ...
# 239       29.03  5.92
# 240       27.18  2.00
# 241       22.67  2.00
# 242       17.82  1.75
# 243       18.78  3.00


# operation directlly based on index 
# if col not yet exist, add at the end
df['tip_percentage'] = 100* df['tip'] / df['total_bill']

# if col already exist, will just 'Update'
df['price_per_person'] = df['total_bill'] / df['size']



# # np.round(a, decimals=0, out=NONE)
# help(np.round)
# Because pandas is based on numpy, we get awesome capabilities with numpy's universal functions!
df['price_per_person'] = np.round(df['price_per_person'],2)



# df.drop() -> shape(row, col)
# removing rows axis=0
# removing columns axis=1
# df.drop('tip_percentage',axis=1)
df = df.drop("tip_percentage",axis=1)


# to remove permently, #1: inplace= , #2: update original df 
df.drop(inplace=True)
df = df.drop("tip_percentage",axis=1)


###############################################################################################################################
""" 
# print(df.set_index('Payment ID'))
# print(df.reset_index())
# df.iloc[0:4] -> multiple rows 
"""


print(df.head())

# total_bill   tip     sex smoker  day    time  size  price_per_person          Payer Name         CC Number Payment ID
# 0       16.99  1.01  Female     No  Sun  Dinner     2              8.49  Christy Cunningham  3560325168603410    Sun2959
# 1       10.34  1.66    Male     No  Sun  Dinner     3              3.45      Douglas Tucker  4478071379779230    Sun4608
# 2       21.01  3.50    Male     No  Sun  Dinner     3              7.00      Travis Walters  6011812112971322    Sun4458
# 3       23.68  3.31    Male     No  Sun  Dinner     2             11.84    Nathaniel Harris  4676137647685994    Sun5260
# 4       24.59  3.61  Female     No  Sun  Dinner     4              6.15        Tonya Carter  4832732618637221    Sun2251



print(df.index)

# RangeIndex(start=0, stop=244, step=1)



# use col name as index and lineup - not affect in place 
print(df.set_index('Payment ID'))

#             total_bill   tip     sex smoker   day    time  size  price_per_person          Payer Name         CC Number
# Payment ID                                                                                                             
# Sun2959          16.99  1.01  Female     No   Sun  Dinner     2              8.49  Christy Cunningham  3560325168603410
# Sun4608          10.34  1.66    Male     No   Sun  Dinner     3              3.45      Douglas Tucker  4478071379779230
# Sun4458          21.01  3.50    Male     No   Sun  Dinner     3              7.00      Travis Walters  6011812112971322
# Sun5260          23.68  3.31    Male     No   Sun  Dinner     2             11.84    Nathaniel Harris  4676137647685994
# Sun2251          24.59  3.61  Female     No   Sun  Dinner     4              6.15        Tonya Carter  4832732618637221
# ...                ...   ...     ...    ...   ...     ...   ...               ...                 ...               ...
# Sat2657          29.03  5.92    Male     No   Sat  Dinner     3              9.68       Michael Avila  5296068606052842
# Sat1766          27.18  2.00  Female    Yes   Sat  Dinner     2             13.59      Monica Sanders  3506806155565404
# Sat3880          22.67  2.00    Male    Yes   Sat  Dinner     2             11.34          Keith Wong  6011891618747196
# Sat17            17.82  1.75    Male     No   Sat  Dinner     2              8.91        Dennis Dixon     4375220550950
# Thur672          18.78  3.00  Female     No  Thur  Dinner     2              9.39     Michelle Hardin  3511451626698139

# [244 rows x 10 columns]




# reset it back to orginal
df = df.reset_index()




# # grab row by index/ labels
# Integer Based
print(df.iloc[0])

# total_bill                       16.99
# tip                               1.01
# sex                             Female
# smoker                              No
# day                                Sun
# time                            Dinner
# size                                 2
# price_per_person                  8.49
# Payer Name          Christy Cunningham
# CC Number             3560325168603410
# Payment ID                     Sun2959
# Name: 0, dtype: object


# Name Based
print(df.loc['Sun2959'])

# # # total_bill                       16.99
# # # tip                               1.01
# # # sex                             Female
# # # smoker                              No
# # # day                                Sun
# # # time                            Dinner
# # # size                                 2
# # # price_per_person                  8.49
# # # Payer Name          Christy Cunningham
# # # CC Number             3560325168603410
# # # Payment ID                     Sun2959
# # # Name: 0, dtype: object




df.iloc[0:4]
df.loc[['Sun2959','Sun5260']]       # pass as a list 

###############################################################################################################################
""" drop rows 
df.drop('Sun2959',axis=0).head()

consider selecting what really want 
df.iloc[1:]

"""



""" insert a now row 
df.append(one_row)
"""

one_row = df.iloc[0]

df.append(one_row)
# print(df)






