
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
""" pre processing data """

# print(df.head())
# #    loan_amnt        term  int_rate  ...  pub_rec_bankruptcies                                            address loan_repaid
# # 0    10000.0   36 months     11.44  ...                   0.0     0174 Michelle Gateway\r\nMendozaberg, OK 22690           1
# # 1     8000.0   36 months     11.99  ...                   0.0  1076 Carney Fort Apt. 347\r\nLoganmouth, SD 05113           1
# # 2    15600.0   36 months     10.49  ...                   0.0  87025 Mark Dale Apt. 269\r\nNew Sabrina, WV 05113           1
# # 3     7200.0   36 months      6.49  ...                   0.0            823 Reid Ford\r\nDelacruzside, MA 00813           1
# # 4    24375.0   60 months     17.27  ...                   0.0             679 Luna Roads\r\nGreggshire, VA 11650           0


# print(len(df))          # 396030


# # check for missing data 
# print(df.isnull().sum())
# # loan_amnt                   0
# # term                        0
# # int_rate                    0
# # installment                 0
# # grade                       0
# # sub_grade                   0
# # emp_title               22927
# # emp_length              18301
# # home_ownership              0
# # annual_inc                  0
# # verification_status         0
# # issue_d                     0
# # loan_status                 0
# # purpose                     0
# # title                    1755
# # dti                         0
# # earliest_cr_line            0
# # open_acc                    0
# # pub_rec                     0
# # revol_bal                   0
# # revol_util                276
# # total_acc                   0
# # initial_list_status         0
# # application_type            0
# # mort_acc                37795
# # pub_rec_bankruptcies      535
# # address                     0
# # loan_repaid                 0
# # dtype: int64


# print(100* df.isnull().sum()/len(df))


""" How many unique employment job titles are there? """
feat_info('emp_title')  
# The job title supplied by the Borrower when applying for the loan.*

print('\n')
feat_info('emp_length')
# Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.



""" To check unique items """
df['emp_title'].nunique()           # 173105

df['emp_title'].value_counts()
# print(df['emp_title'].value_counts())
# # Teacher                                     4389
# # Manager                                     4250
# # Registered Nurse                            1856
# # RN                                          1846
# # Supervisor                                  1830
# #                                             ... 
# # Lender Specialist                              1
# # NEC Laboratories America, Inc                  1
# # Sr. Nuclear Health Physicist                   1
# # Texas Health Center for Diagnostics and        1
# # United Stated Dept of Treasury, CC Offic       1
# # Name: emp_title, Length: 173105, dtype: int64





""" Realistically there are too many unique job titles to try to convert this to a dummy variable feature. 
Let's remove that emp_title column. """ 
df = df.drop('emp_title', axis=1)



""" Create a count plot of the emp_length feature column. Challenge: Sort the order of the values """ 
sorted(df['emp_length'].dropna().unique())

# print(sorted(df['emp_length'].dropna().unique()))
emp_length_order = [ '< 1 year',
                    '1 year','2 years','3 years','4 years','5 years',
                    '6 years','7 years','8 years','9 years','10+ years'
                    ]


plt.figure('emp_length with ordered',figsize=(12,4))
sns.countplot(x='emp_length', data=df, order=emp_length_order)
plt.show()



""" Plot out the countplot with a hue separating Fully Paid vs Charged Off"""
plt.figure('emp length vs loan status',figsize=(12,4))
sns.countplot(x='emp_length', data=df, order=emp_length_order, hue='loan_status')
# plt.show()


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length', data=df, order=emp_length_order)
# plt.show()



""" Plot out the countplot with a hue separating Fully Paid vs Charged Off """ 
plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')
plt.show()



""" This still doesn't really inform us if there is a strong relationship between employment length and being charged off, 
what we want is the percentage of charge offs per category. 
Essentially informing us what percent of people per employment category didn't pay back their loan. 
There are a multitude of ways to create this Series. 
Once you've created it, see if visualize it with a [bar plot](https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.plot.html). 
This may be tricky, refer to solutions if you get stuck on creating this Series """ 

emp_co = df[df['loan_status'] == "Charged Off"].groupby("emp_length").count()['loan_status']

emp_fp = df[df['loan_status'] == "Fully Paid"].groupby("emp_length").count()['loan_status']

emp_len = emp_co/emp_fp
# print(emp_len)
# # emp_length
# # 1 year       0.248649
# # 10+ years    0.225770
# # 2 years      0.239560
# # 3 years      0.242593
# # 4 years      0.238213
# # 5 years      0.237911
# # 6 years      0.233341
# # 7 years      0.241887
# # 8 years      0.249625
# # 9 years      0.250735
# # < 1 year     0.260830
# # Name: loan_status, dtype: float64

emp_len.plot(kind='bar')
# plt.show()


""" Revisit the DataFrame to see what feature columns still have missing data. """ 
df = df.drop('emp_length',axis=1)
# print(df.isnull().sum())
# # loan_amnt                   0
# # term                        0
# # int_rate                    0
# # installment                 0
# # grade                       0
# # sub_grade                   0
# # home_ownership              0
# # annual_inc                  0
# # verification_status         0
# # issue_d                     0
# # loan_status                 0
# # purpose                     0
# # title                    1755
# # dti                         0
# # earliest_cr_line            0
# # open_acc                    0
# # pub_rec                     0
# # revol_bal                   0
# # revol_util                276
# # total_acc                   0
# # initial_list_status         0
# # application_type            0
# # mort_acc                37795
# # pub_rec_bankruptcies      535
# # address                     0
# # loan_repaid                 0
# # dtype: int64

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
"""" deal with missing data """

# # check repeated information
# print(df['purpose'].head(10))
# feat_info('purpose')
# print(df['title'].head(10))


""" The title column is simply a string subcategory/description of the purpose column. 
Go ahead and drop the title column.
""" 
df = df.drop('title',axis=1)


""" This is one of the hardest parts of the project! Refer to the solutions video if you need guidance, 
feel free to fill or drop the missing values of the mort_acc however you see fit! 
Here we're going with a very specific approach.

Find out what the mort_acc feature represents
""" 
feat_info('mort_acc')
# print(df['mort_acc'].value_counts())
# # 0.0     139777
# # 1.0      60416
# # 2.0      49948
# # 3.0      38049

# # show more (open the raw output data in a text editor) ...


# # 28.0         1
# # 34.0         1
# # Name: mort_acc, dtype: int64




print("Correlation with the mort_acc column")
# print(df.corr()['mort_acc'].sort_values())
# # Correlation with the mort_acc column
# # int_rate               -0.082583
# # dti                    -0.025439
# # revol_util              0.007514
# # pub_rec                 0.011552
# # pub_rec_bankruptcies    0.027239
# # loan_repaid             0.073111
# # open_acc                0.109205
# # installment             0.193694
# # revol_bal               0.194925
# # loan_amnt               0.222315
# # annual_inc              0.236320
# # total_acc               0.381072
# # mort_acc                1.000000
# # Name: mort_acc, dtype: float64



""" 
Looks like the total_acc feature correlates with the mort_acc , this makes sense! Let's try this fillna() approach. 
We will group the dataframe by the total_acc and calculate the mean value for the mort_acc per total_acc entry. 
To get the result below:
"""
print("Mean of mort_acc column per total_acc")
# print(df.groupby('total_acc').mean()['mort_acc'])
# # Mean of mort_acc column per total_acc
# # total_acc
# # 2.0      0.000000
# # 3.0      0.052023
# # 4.0      0.066743
# #            ...   
# # 150.0    2.000000
# # 151.0    0.000000
# # Name: mort_acc, Length: 118, dtype: float64



""" 
Let's fill in the missing mort_acc values based on their total_acc value. 
If the mort_acc is missing, then we will fill in that missing value with the mean value corresponding to its total_acc value from the Series we created above. 
This involves using an .apply() method with two columns. Check out the link below for more info, or review the solutions video/notebook.**

[Helpful Link](https://stackoverflow.com/questions/13331698/how-to-apply-a-function-to-two-columns-of-pandas-dataframe) 
"""

total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
# print(total_acc_avg[2.0])           # 0.0

def fill_mort_acc(total_acc, mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
# print(df.isnull().sum())
# # Loan_amnt                 0
# # term                      0
# # int_rate                  0
# # installment               0
# # grade                     0
# # sub_grade                 0
# # home_ownership            0
# # annual_inc                0
# # verification_status       0
# # issue_d                   0
# # loan_status               0
# # purpose                   0
# # dti                       0
# # earliest_cr_line          0
# # open_acc                  0
# # pub_rec                   0
# # revol_bal                 0
# # revol_util              276
# # total_acc                 0
# # initial_list_status       0
# # application_type          0
# # mort_acc                  0
# # pub_rec_bankruptcies    535
# # address                   0
# # loan_repaid               0
# # dtype: int64












plt.close()


















