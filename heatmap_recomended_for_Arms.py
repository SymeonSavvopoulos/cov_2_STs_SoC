import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_A = pd.read_csv(".../arm_A_5d_60d.csv")
data_B = pd.read_csv(".../arm_B_5d_60d.csv")

data_A=data_A.iloc[:,:-1]
data_B=data_B.iloc[:,:-1]


corr_A = pd.DataFrame(data_A).corr()
corr_B = pd.DataFrame(data_B).corr()

mask_upper = np.triu(np.ones_like(corr_A, dtype=bool), 1)  
mask_lower = np.tril(np.ones_like(corr_A, dtype=bool), -1) 

combined_data = corr_A.where(mask_lower, 0) + corr_B.where(mask_upper, 0) + np.diag(np.diag(corr_A))

# Plotting the heatmap
plt.figure(figsize=(25, 16))
ax = sns.heatmap(combined_data, annot=True, cmap='coolwarm', square=True, linewidths=.5,
                 annot_kws={"size": 12, "weight": "bold"}) 

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, horizontalalignment='right', fontsize=18)



threshold = 0.7  
for i in range(combined_data.shape[0]):
    for j in range(combined_data.shape[1]):
        value = combined_data.iloc[i, j]
        if abs(value) > threshold and abs(value) < 0.99:
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor='black', lw=10))

plt.title('Heatmap with Arm A in Lower and Arm B in Upper Triangles')
plt.show()
