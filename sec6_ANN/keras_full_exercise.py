
import pandas as pd


data_info = pd.read_csv('./DATA/lending_club_info.csv',index_col='LoanStatNew')
# print(data_info.head())

# print(data_info.loc['revol_util']['Description'])
# # Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

# feat_info('mort_acc')           # Number of mortgage accounts.


###################################################################################################################################
###################################################################################################################################

""" loading data set """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./DATA/lending_club_loan_two.csv')
# print(df.info())
# # RangeIndex: 396030 entries, 0 to 396029
# # Data columns (total 27 columns):
# #  #   Column                Non-Null Count   Dtype  
# # ---  ------                --------------   -----  
# #  0   loan_amnt             396030 non-null  float64
# #  1   term                  396030 non-null  object 
# #  2   int_rate              396030 non-null  float64
# #  3   installment           396030 non-null  float64
# #  4   grade                 396030 non-null  object 
# #  5   sub_grade             396030 non-null  object 
# #  6   emp_title             373103 non-null  object 
# #  7   emp_length            377729 non-null  object 
# #  8   home_ownership        396030 non-null  object 
# #  9   annual_inc            396030 non-null  float64
# #  10  verification_status   396030 non-null  object 
# #  11  issue_d               396030 non-null  object 
# #  12  loan_status           396030 non-null  object 
# #  13  purpose               396030 non-null  object 
# #  14  title                 394275 non-null  object 
# #  15  dti                   396030 non-null  float64
# #  16  earliest_cr_line      396030 non-null  object 
# #  17  open_acc              396030 non-null  float64
# #  18  pub_rec               396030 non-null  float64
# #  19  revol_bal             396030 non-null  float64
# #  20  revol_util            395754 non-null  float64
# #  21  total_acc             396030 non-null  float64
# #  22  initial_list_status   396030 non-null  object 
# #  23  application_type      396030 non-null  object 
# #  24  mort_acc              358235 non-null  float64
# #  25  pub_rec_bankruptcies  395495 non-null  float64
# #  26  address               396030 non-null  object 
# # dtypes: float64(12), object(15)
# # memory usage: 81.6+ MB
# # None


# print(df.head())
# #    loan_amnt        term  int_rate  ...  mort_acc pub_rec_bankruptcies                                            address
# # 0    10000.0   36 months     11.44  ...       0.0                  0.0     0174 Michelle Gateway\r\nMendozaberg, OK 22690
# # 1     8000.0   36 months     11.99  ...       3.0                  0.0  1076 Carney Fort Apt. 347\r\nLoganmouth, SD 05113
# # 2    15600.0   36 months     10.49  ...       0.0                  0.0  87025 Mark Dale Apt. 269\r\nNew Sabrina, WV 05113
# # 3     7200.0   36 months      6.49  ...       0.0                  0.0            823 Reid Ford\r\nDelacruzside, MA 00813
# # 4    24375.0   60 months     17.27  ...       1.0                  0.0             679 Luna Roads\r\nGreggshire, VA 11650

# print(df.columns)


plt.figure('loan status')
sns.countplot(x='loan_status', data=df)
# plt.show()


plt.figure('loan amount', figsize=(12,4))
sns.distplot(df['loan_amnt'], kde=False, bins=40)
plt.xlim(0,45000)
# plt.show()


# print(df.corr())
# #                       loan_amnt  int_rate  installment  annual_inc       dti  ...  revol_bal  revol_util  total_acc  mort_acc  pub_rec_bankruptcies
# # loan_amnt              1.000000  0.168921     0.953929    0.336887  0.016636  ...   0.328320    0.099911   0.223886  0.222315             -0.106539
# # int_rate               0.168921  1.000000     0.162758   -0.056771  0.079038  ...  -0.011280    0.293659  -0.036404 -0.082583              0.057450
# # installment            0.953929  0.162758     1.000000    0.330381  0.015786  ...   0.316455    0.123915   0.202430  0.193694             -0.098628
# # annual_inc             0.336887 -0.056771     0.330381    1.000000 -0.081685  ...   0.299773    0.027871   0.193023  0.236320             -0.050162
# # dti                    0.016636  0.079038     0.015786   -0.081685  1.000000  ...   0.063571    0.088375   0.102128 -0.025439             -0.014558
# # open_acc               0.198556  0.011649     0.188973    0.136150  0.136181  ...   0.221192   -0.131420   0.680728  0.109205             -0.027732
# # pub_rec               -0.077779  0.060986    -0.067892   -0.013720 -0.017639  ...  -0.101664   -0.075910   0.019723  0.011552              0.699408
# # revol_bal              0.328320 -0.011280     0.316455    0.299773  0.063571  ...   1.000000    0.226346   0.191616  0.194925             -0.124532
# # revol_util             0.099911  0.293659     0.123915    0.027871  0.088375  ...   0.226346    1.000000  -0.104273  0.007514             -0.086751
# # total_acc              0.223886 -0.036404     0.202430    0.193023  0.102128  ...   0.191616   -0.104273   1.000000  0.381072              0.042035
# # mort_acc               0.222315 -0.082583     0.193694    0.236320 -0.025439  ...   0.194925    0.007514   0.381072  1.000000              0.027239
# # pub_rec_bankruptcies  -0.106539  0.057450    -0.098628   -0.050162 -0.014558  ...  -0.124532   -0.086751   0.042035  0.027239              1.000000

# # [12 rows x 12 columns]


""" 
heatmap: https://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap 
resizing: https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot 
"""
plt.figure('heatmap', figsize=(12,7))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.ylim(10, 0)
# plt.show()

""" 
You should have noticed almost perfect correlation with the "installment" feature. 
Explore this feature further. Print out their descriptions and perform a scatterplot between them. 
Does this relationship make sense to you? Do you think there is duplicate information here?
"""

# print(feat_info('installment'))
# # The monthly payment owed by the borrower if the loan originates.

# print(feat_info('loan_amnt'))
# # The listed amount of the loan applied for by the borrower. 
# # If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.


plt.figure('installement VS loan amount',figsize=(12,7))
sns.scatterplot(x='installment', y='loan_amnt', data=df)
# plt.show()



# Create a boxplot showing the relationship between the loan_status and the Loan Amount
plt.figure('loan status VS loan amount', figsize=(12,7))
sns.boxplot(x='loan_status', y='loan_amnt', data=df)
# plt.show()


# Calculate the summary statistics for the loan amount, grouped by the loan_status
df.groupby('loan_status')['loan_amnt'].describe()
# print(df.groupby('loan_status')['loan_amnt'].describe())
# #                 count          mean          std     min     25%      50%      75%      max
# # loan_status                                                                                
# # Charged Off   77673.0  15126.300967  8505.090557  1000.0  8525.0  14000.0  20000.0  40000.0
# # Fully Paid   318357.0  13866.878771  8302.319699   500.0  7500.0  12000.0  19225.0  40000.0



# TASK: Let's explore the Grade and SubGrade columns that LendingClub attributes to the loans. 
# What are the unique possible grades and subgrades?
sorted(df['grade'].unique())
sorted(df['sub_grade'].unique())



# Create a countplot per grade. Set the hue to the loan_status label
plt.figure('grade VS loan status', figsize=(12,7))
sns.countplot(x='grade', data=df, hue='loan_status')
plt.show()



# TASK: Display a count plot per subgrade. You may need to resize for this plot and reorder the x axis. 
# Feel free to edit the color palette. 
# Explore both all loans made per subgrade as well being separated based on the loan_status
plt.figure('subgrade',figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order = subgrade_order, palette='coolwarm')
plt.show()


plt.figure('subgrade with hue ',figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order = subgrade_order, palette='coolwarm', hue='loan_status')
plt.show()



# It looks like F and G subgrades don't get paid back that often. 
# Isloate those and recreate the countplot just for those subgrades
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')
plt.show()



# Create a new column called 'load_repaid' which will contain a 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off"
df['loan_status'].unique()
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1, 'Charged Off':0})
df[['loan_repaid','loan_status']]

# (Note this is hard, but can be done in one line!) 
# Create a bar plot showing the correlation of the numeric features to the new loan_repaid column. 
# [Helpful Link](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html)
plt.figure('loan repaid corr', figsize=(12,7))
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.show()

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################


















