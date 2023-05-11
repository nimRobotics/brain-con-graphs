import numpy as np
import pandas as pd


#  load the csv file
df = pd.read_csv('./input/funcCon.csv')
print(df.head())

# store the information from the first 4 columns
df_info = df.iloc[:, 0:4]
print(df_info.head())

# 
df_data = df.iloc[:, 4:]
print(df_data.head())

# store the information from the first 4 columns
for i in range(0, df_data.shape[0]):
    con_mat = df_data.iloc[i, :].to_numpy()
    zscores = np.arctanh(con_mat)
    con_mat[np.abs(zscores) < 0.4] = 0
    non_zero_con = np.count_nonzero(con_mat)
    print('row {} has {} non-zero connections'.format(i, non_zero_con))
    df_info.loc[i, 'non_zero_con'] = non_zero_con

# add Reliability column, 0 if condtion contains 'HR', 1 if condition contains 'LR'
df_info['Reliability'] = df_info['condition'].apply(lambda x: 0 if 'HR' in x else 1)
# add Fatigue column, 0 if condtion contains 'N', 1 if condition contains 'F'
df_info['Fatigue'] = df_info['condition'].apply(lambda x: 0 if 'N' in x else 1)

print(df_info.head())
# save to csv
df_info.to_csv('./output/fc_no_of_con.csv', index=False)
