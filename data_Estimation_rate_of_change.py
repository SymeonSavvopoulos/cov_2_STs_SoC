import pandas as pd
import numpy as np
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu
import seaborn as sns
import matplotlib.pyplot as plt

dfA0 = pd.read_csv(".../arm_A_0d_60d.csv",header=0)
dfA5 = pd.read_csv(".../arm_A_5d_60d.csv",header=0)



dfA0=dfA0.iloc[:,4:]
dfA5=dfA5.iloc[:,4:-1]

dfrA=(dfA5-dfA0)/5

dfB0 = pd.read_csv(".../arm_B_0d_60d.csv")
dfB5 = pd.read_csv(".../arm_B_5d_60d.csv")

dfB0=dfB0.iloc[:,4:]
dfB5=dfB5.iloc[:,4:-1]

dfrB=(dfB5-dfB0)/5


dfA0['Y'] = 0
dfB0['Y'] = 1
d0=pd.concat((dfA0,dfB0),ignore_index=True)



dfrA['Y'] = 0
dfrB['Y'] = 1
dr0=pd.concat((dfrA,dfrB),ignore_index=True)




def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    df_cleaned = df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]
    return df_cleaned

features = d0.columns[:-1]
target = d0.columns[-1]

d0_cleaned = remove_outliers(d0, features)

d0_melted = pd.melt(d0_cleaned, id_vars=target, var_name='Feature', value_name='Value')

plt.figure(figsize=(17, 8))

sns.violinplot(data=d0_melted, x='Feature', y='Value', hue=target, split=False, inner='box', linewidth=1.5, cut=0)

plt.title('Violin Plot of Features by Class (without outliers)', fontsize=16)
plt.ylabel('Values of the biomarkers at baseline', fontsize=19)
plt.xlabel('Biomarkers', fontsize=19)
categories = ['Cov-2-NAbs', 'IFNg','IL10','IL2','IL6','TNFa','CD3','CD4','CD56','CD8','CoV-2-STs','CPK','CRP','DD','Ferritin','LDH', 'WBC']
#plt.plot(categories, values, marker='o')  # Simple line plot with markers

# Setting x-axis labels
plt.xticks(ticks=range(len(categories)), labels=categories)
plt.xticks(rotation=90, fontsize=19)
plt.yticks(fontsize=19)

# Adjust the legend
plt.legend(title=target, title_fontsize='13', fontsize='15', loc='lower left', bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()
plt.show()



# Unique classes in the target
classes0 = d0[target].unique()
if len(classes0) != 2:
    raise ValueError("Mann-Whitney U test requires exactly two groups.")

# Perform Mann-Whitney U test for each feature
results0 = {}

for feature in features:
    group10 = d0[d0[target] == classes0[0]][feature].values
    group20 = d0[d0[target] == classes0[1]][feature].values
    stat, p_value = mannwhitneyu(group10, group20, alternative='two-sided')
    results0[feature] = {'statistic': stat, 'p_value': p_value}

# Convert results to a DataFrame for easy viewing
results_d0 = pd.DataFrame(results0).T

# Display the results
print(results_d0)




features = dr0.columns[:-1]
target = dr0.columns[-1]

dr0_cleaned = remove_outliers(dr0, features)

dr0_melted = pd.melt(dr0_cleaned, id_vars=target, var_name='Feature', value_name='Value')

plt.figure(figsize=(17, 8))

sns.violinplot(data=dr0_melted, x='Feature', y='Value', hue=target, split=False, inner='box', linewidth=1.5, cut=0)

plt.title('Violin Plot of Features by Class (without outliers)', fontsize=16)
plt.ylabel('Biomarkers rate of change at 5th day', fontsize=19)
plt.xlabel('Biomarkers', fontsize=19)
plt.xticks(ticks=range(len(categories)), labels=categories)
plt.xticks(rotation=90, fontsize=19)
plt.yticks(fontsize=19)

# Adjust the legend
plt.legend(title=target, title_fontsize='13', fontsize='15', loc='lower left', bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()
plt.show()



# Unique classes in the target
classes0 = dr0[target].unique()
if len(classes0) != 2:
    raise ValueError("Mann-Whitney U test requires exactly two groups.")

# Perform Mann-Whitney U test for each feature
results0 = {}

for feature in features:
    group10 = dr0[d0[target] == classes0[0]][feature].values
    group20 = dr0[d0[target] == classes0[1]][feature].values
    stat, p_value = mannwhitneyu(group10, group20, alternative='two-sided')
    results0[feature] = {'statistic': stat, 'p_value': p_value}

# Convert results to a DataFrame for easy viewing
results_dr0 = pd.DataFrame(results0).T

# Display the results
print(results_dr0)