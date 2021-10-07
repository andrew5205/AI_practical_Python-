
import numpy as np 
import pandas as pd 


# help(pd.Series())
# https://pandas.pydata.org/docs/reference/api/pandas.Series.html


pd.Series()

my_index = ['USA', 'Canada', 'Mexico']
my_data = [1776, 1867, 1821]

my_series = pd.Series(data=my_data)
# print(my_series)

# # 0    1776
# # 1    1867
# # 2    1821
# # dtype: int64


# print(my_series[0])         # 1776


my_series = pd.Series(data=my_data, index=my_index)
# print(my_series)

# # USA       1776
# # Canada    1867
# # Mexico    1821
# # dtype: int64


# print(my_series['USA'])         # 1776
#############################################################################


ages = {'Sam':5, 'Frank':10, 'Spike':7}

age_series = pd.Series(ages)
# print(age_series)

# # Sam       5
# # Frank    10
# # Spike     7
# # dtype: int64





#############################################################################
""" operation """
# Imaginary Sales Data for 1st and 2nd Quarters for Global Company
q1 = {'Japan': 80, 'China': 450, 'India': 200, 'USA': 250}
q2 = {'Brazil': 100,'China': 500, 'India': 210,'USA': 260}


# Convert into Pandas Series
sales_Q1 = pd.Series(q1)
sales_Q2 = pd.Series(q2)



# Call values based on Named Index
sales_Q1['Japan']
# print(sales_Q1['Japan'])        # 80

# Integer Based Location information also retained!
sales_Q1[0]
# print(sales_Q1[0])          # 80



# Grab just the index keys
sales_Q1.keys()
# print(sales_Q1.keys())          # Index(['Japan', 'China', 'India', 'USA'], dtype='object')



# Can Perform Operations Broadcasted across entire Series
sales_Q1 * 2
# print(sales_Q1 * 2)

# # Japan    160
# # China    900
# # India    400
# # USA      500
# # dtype: int64



# Notice how Pandas informs you of mismatch with NaN
merge = sales_Q1 + sales_Q2
# print(merge)

# # Brazil      NaN
# # China     950.0
# # India     410.0
# # Japan       NaN
# # USA       510.0
# # dtype: float64


# You can fill these with any value you want
fill_value = sales_Q1.add(sales_Q2,fill_value=0)
# print(fill_value)

# # Brazil    100.0
# # China     950.0
# # India     410.0
# # Japan      80.0
# # USA       510.0




