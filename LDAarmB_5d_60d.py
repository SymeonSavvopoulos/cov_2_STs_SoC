# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 23:08:36 2022

@author: simon
"""


"""
Created on Thu Mar 24 20:11:29 2022

@author: simon
"""

import numpy as np


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)


        mean_overall = np.mean(X, axis=0)
        SW = np.zeros((n_features, n_features))
        SB = np.zeros((n_features, n_features))
        for c in class_labels:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            SW += (X_c - mean_c).T.dot((X_c - mean_c))
            n_c = X_c.shape[0]
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            SB += n_c * (mean_diff).dot(mean_diff.T)


        A = np.linalg.inv(SW).dot(SB)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        
        eigenvectors = eigenvectors.T
        idxs = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
     
        self.linear_discriminants = eigenvectors[0 : self.n_components]

    def transform(self, X):
     
        return np.dot(X, self.linear_discriminants.T)


# Testing
if __name__ == "__main__":
    # Imports
    import csv
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
    
    
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import precision_score, classification_report
    
    data= pd.read_csv(".../arm_B_5d_60d.csv")



    
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    
    correlations = data.corr()
    f, ax= plt.subplots(figsize=(20,20))
    sns.heatmap(correlations,annot=True)
    
    

    X = data[['age', 'gender', 'karn', 'corm', 'con2Abs', 'IFNg', 'IL10','IL2','IL6','TNFa',
              'CD3','CD4', 'CD56','CD8', 'CoV_2_STs', 'CPK', 'CRP', 'DD',
                'Ferrit', 'LDH', 'WBC']]
    
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X=sc.fit_transform(X)


    
    lda = LDA(2)
    lda.fit(X, y)
    
    X_projected = lda.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1, x2 = X_projected[:, 0], X_projected[:, 1]

    plt.scatter(x1.real, x2.real, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
    
    

    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
    plt.colorbar()
    plt.show()
    
    scatter = plt.scatter(x1.real, x2.real, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))

    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")
#    plt.colorbar()

    from matplotlib.font_manager import FontProperties
    # Add legend
    font = FontProperties(family='Times New Roman', weight='bold',size=12)
    font.set_weight('bold')
    
    plt.xlabel("Linear Discriminant 1", fontproperties=font)
    plt.ylabel("Linear Discriminant 2", fontproperties=font)
    plt.title("Scatter Plot of Linear Discriminants", fontproperties=font)
    classes = ['Alive and well', 'Died', 'Alive with disease']
    plt.legend(handles=scatter.legend_elements()[0], title="Classes", labels=classes, prop=font)
    
    plt.show()
    
    
    
   
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_projected.real, y, test_size = 0.3, random_state=10)
    
    
    
    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train.real, y_train)
    
    
    scatter = plt.scatter(x1.real, x2.real, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))

    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")


    from matplotlib.font_manager import FontProperties

    font = FontProperties(family='Times New Roman', weight='bold')
    font.set_weight('bold')
    
    plt.xlabel("Linear Discriminant 1", fontproperties=font)
    plt.ylabel("Linear Discriminant 2", fontproperties=font)
    plt.title("Scatter Plot of Linear Discriminants", fontproperties=font)
    classes = ['Alive and well', 'Died', 'Alive with disease']
    plt.legend(handles=scatter.legend_elements()[0], title="Classes", labels=classes, prop=font)
    
    plt.show()

    

    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_projected.real, y, test_size = 0.3, random_state=10)
    
    
    
    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train.real, y_train)
    
    

    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.model_selection import cross_val_score
    lin_score=cross_val_score(classifier,X_projected.real,y,cv=3,scoring='accuracy')
    y_pred = cross_val_predict(classifier, X_projected.real, y, cv=3)
    

    precision_per_class = precision_score(y, y_pred, average=None)
    

    print("Precision per class:")
    for i, precision in enumerate(precision_per_class):
        print(f"Class {i}: {precision:.4f}")
    

    macro_precision = precision_score(y, y_pred, average='macro')
    micro_precision = precision_score(y, y_pred, average='micro')
    
    print(f"\nMacro Average Precision: {macro_precision:.4f}")
    print(f"Micro Average Precision: {micro_precision:.4f}")
    
    cv = StratifiedKFold(n_splits=3)
    
    precision_per_fold = []
    

    precision_per_fold = []
   

    for train_index, test_index in cv.split(X, y):
       X_train, X_test = X_projected.real[train_index], X_projected.real[test_index]
       y_train, y_test = y[train_index], y[test_index]
   
      
       y_pred = classifier.predict(X_test)
      

       precision_fold = precision_score(y_test, y_pred, average=None)
       precision_per_fold.append(precision_fold)
   
    precision_per_fold = np.array(precision_per_fold)
   
   
    std_dev_per_class = np.std(precision_per_fold, axis=0)
   
   
    print("Standard Deviations of Precision per class:")
    for i, std_dev in enumerate(std_dev_per_class):
        print(f"Class {i}: {std_dev:.4f}")
   
    
    
    
   
    
    x1, x2 = X_projected[:, 0], X_projected[:, 1]

    plt.scatter(x1.real, x2.real, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))
    
    h = 0.02  
    x_min, x_max = X_projected[:, 0].min() - 1, X_projected[:, 0].max() + 1
    y_min, y_max = X_projected[:, 1].min() - 1, X_projected[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min.real, x_max.real, h), np.arange(y_min.real, y_max.real, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    

    plt.contour(xx, yy, Z, colors='k', linewidths=1, alpha=0.5)
    
    
    scatter = plt.scatter(x1.real, x2.real, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3))

    plt.tick_params(axis='both', which='both', direction='out', width=2, length=6, labelsize=12, labelcolor='black', pad=10)

    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.xlabel("Linear Discriminant 1")
    plt.ylabel("Linear Discriminant 2")


    from matplotlib.font_manager import FontProperties

    font = FontProperties(family='Times New Roman', weight='bold')
    font.set_weight('bold')
    
    plt.xlabel("Linear Discriminant 1", fontproperties=font)
    plt.ylabel("Linear Discriminant 2", fontproperties=font)
    plt.title("Scatter Plot of Linear Discriminants", fontproperties=font)
    classes = ['Alive and well', 'Died', 'Alive with disease']
    plt.legend(handles=scatter.legend_elements()[0], title="Classes",labels=classes, prop=font)
    
    plt.show()
    


    n_features = X.shape[1]
    class_labels = np.unique(y)



    mean_overall = np.mean(X, axis=0)
    SW = np.zeros((n_features, n_features))
    SB = np.zeros((n_features, n_features))
    for c in class_labels:
        X_c = X[y == c]
        mean_c = np.mean(X_c, axis=0)

        SW += (X_c - mean_c).T.dot((X_c - mean_c))


        n_c = X_c.shape[0]
        mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
        SB += n_c * (mean_diff).dot(mean_diff.T)


    A = np.linalg.inv(SW).dot(SB)

    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    eigenvectors = eigenvectors.T
    idxs = np.argsort(abs(eigenvalues))[::-1]
    eigenvalues = eigenvalues[idxs]
    eigenvectors = eigenvectors[idxs]
    eigenvectors=eigenvectors.real
    eigenvectors=abs(eigenvectors)

    data0=data.iloc[:, :-1]
    names = data0.columns.tolist()
    eigenvectors1 = eigenvectors[0, :].tolist()

    combined_df1 =  pd.DataFrame({
        'Name': names,
        'Eigenvector': eigenvectors1
        })


    transposed_df1 = combined_df1.T


    sorted_df1 = combined_df1.sort_values(by='Eigenvector', ascending=False)

    eigenvectors2 = eigenvectors[1, :].tolist()

    combined_df2 =  pd.DataFrame({
        'Name': names,
        'Eigenvector': eigenvectors2
        })


    transposed_df2 = combined_df2.T


    sorted_df2 = combined_df2.sort_values(by='Eigenvector', ascending=False)

    

import pandas as pd
from scipy.stats import kruskal





features = data.columns[:-1]
target = data.columns[-1]


results = {}

for feature in features:
    groups = [data[data[target] == cls][feature].values for cls in data[target].unique()]
    stat, p_value = kruskal(*groups)
    results[feature] = {'statistic': stat, 'p_value': p_value}


resultsKW_df = pd.DataFrame(results).T


print(resultsKW_df)



from sklearn.metrics import f1_score
   
  
f1_scores_per_fold = []

for train_index, test_index in cv.split(X_projected.real, y):
    X_train_fold, X_test_fold = X_projected.real[train_index], X_projected.real[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    classifier.fit(X_train_fold, y_train_fold)
    y_pred_fold = classifier.predict(X_test_fold)

    # Calculate F1 score for the current fold
    f1_score_fold = f1_score(y_test_fold, y_pred_fold, average='macro')
    f1_scores_per_fold.append(f1_score_fold)

# Calculate the average F1 score across all folds
average_f1_score = np.mean(f1_scores_per_fold)
print(f"Average F1 Score (Macro): {average_f1_score:.4f}")

import scipy.stats as stats
import seaborn as sns


df=data

df = df.sort_values('Y_60', ascending=True)


from scipy.stats import chi2_contingency


class_mapping = {0: "Alive and well", 1: "Deceased", 2: "Alive with disease"}
df['Y_60'] = df['Y_60'].map(class_mapping)





def get_star_significance(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'






kruskal_result = stats.kruskal(
    df[df['Y_60'] == 'Alive and well']['CD3'],
    df[df['Y_60'] == 'Deceased']['CD3'],
    df[df['Y_60'] == 'Alive with disease']['CD3']
)


pairwise_tests = {}
combinations = [('Alive and well', 'Deceased'), ('Alive and well', 'Alive with disease'), ('Deceased', 'Alive with disease')]
for combo in combinations:
    result = stats.mannwhitneyu(
        df[df['Y_60'] == combo[0]]['CD3'],
        df[df['Y_60'] == combo[1]]['CD3'],
        alternative='two-sided'
    )
    pairwise_tests[combo] = result.pvalue



sns.set(font_scale=1.5)  


plt.figure(figsize=(10, 6))
sns.boxplot(x='Y_60', y='CD3', data=df, showfliers=False)  
plt.title(f'Kruskal-Wallis H-test: H={kruskal_result.statistic:.2f}, p={kruskal_result.pvalue:.3e}', fontsize=16)


positions = [(0, 1)]

height_increments = [500, 200, 10]
for pos, combo, increment in zip(positions, combinations, height_increments):
    y_max =max(df[df['Y_60'] == combo[0]]['CD3'].max(), df[df['Y_60'] == combo[1]]['CD3'].max()) + increment
    stars = get_star_significance(pairwise_tests[combo])
    plt.plot(pos, [y_max, y_max], lw=0.8, c='black')
    plt.text(sum(pos) / 2, y_max + 20.05, stars, ha='center')

plt.xlabel('')  # Set x-axis label, adjust as necessary
plt.ylabel('CD3 (cells/μL)')  # Set y-axis label, adjust as necessary
plt.show()









kruskal_result = stats.kruskal(
    df[df['Y_60'] == 'Alive and well']['CoV_2_STs'],
    df[df['Y_60'] == 'Deceased']['CoV_2_STs'],
    df[df['Y_60'] == 'Alive with disease']['CoV_2_STs']
)

# Perform pairwise Mann-Whitney U tests
pairwise_tests = {}
combinations = [('Alive and well', 'Deceased'), ('Alive and well', 'Alive with disease'), ('Deceased', 'Alive with disease')]
for combo in combinations:
    result = stats.mannwhitneyu(
        df[df['Y_60'] == combo[0]]['CoV_2_STs'],
        df[df['Y_60'] == combo[1]]['CoV_2_STs'],
        alternative='two-sided'
    )
    pairwise_tests[combo] = result.pvalue




sns.set(font_scale=1.5)  


plt.figure(figsize=(10, 6))
sns.boxplot(x='Y_60', y='CoV_2_STs', data=df, showfliers=False)  
plt.title(f'Kruskal-Wallis H-test: H={kruskal_result.statistic:.2f}, p={kruskal_result.pvalue:.3e}', fontsize=16)


positions = [(0, 1)]

height_increments = [100, 100, 100]
for pos, combo, increment in zip(positions, combinations, height_increments):
    y_max =max(df[df['Y_60'] == combo[0]]['CoV_2_STs'].max(), df[df['Y_60'] == combo[1]]['CoV_2_STs'].max()) + increment
    stars = get_star_significance(pairwise_tests[combo])
    plt.plot(pos, [y_max, y_max], lw=0.8, c='black')
    plt.text(sum(pos) / 2, y_max + 0.05, stars, ha='center')

plt.xlabel('')  
plt.ylabel('CoV-2-STs (SFC/5$\cdot$ 10$^5$ PBMCs)')  
plt.show()



kruskal_result = stats.kruskal(
    df[df['Y_60'] == 'Alive and well']['CRP'],
    df[df['Y_60'] == 'Deceased']['CRP'],
    df[df['Y_60'] == 'Alive with disease']['CRP']
)

# Perform pairwise Mann-Whitney U tests
pairwise_tests = {}
combinations = [('Alive and well', 'Deceased'), ('Alive and well', 'Alive with disease'), ('Deceased', 'Alive with disease')]
for combo in combinations:
    result = stats.mannwhitneyu(
        df[df['Y_60'] == combo[0]]['CRP'],
        df[df['Y_60'] == combo[1]]['CRP'],
        alternative='two-sided'
    )
    pairwise_tests[combo] = result.pvalue




sns.set(font_scale=1.5)  

# Plotting
plt.figure(figsize=(10, 6))
sns.boxplot(x='Y_60', y='CRP', data=df, showfliers=False)  
plt.title(f'Kruskal-Wallis H-test: H={kruskal_result.statistic:.2f}, p={kruskal_result.pvalue:.3e}', fontsize=16)


positions = [(0, 1)]

height_increments = [10, 10, 10]
for pos, combo, increment in zip(positions, combinations, height_increments):
    y_max =max(df[df['Y_60'] == combo[0]]['CRP'].max(), df[df['Y_60'] == combo[1]]['CRP'].max()) + increment
    stars = get_star_significance(pairwise_tests[combo])
    plt.plot(pos, [y_max, y_max], lw=0.8, c='black')
    plt.text(sum(pos) / 2, y_max + 0.05, stars, ha='center')

plt.xlabel('')  # Set x-axis label, adjust as necessary
plt.ylabel('CRP (g/dL)')  # Set y-axis label, adjust as necessary
plt.show()




df=data


class_mapping = {0: "Alive and well", 1: "Deceased", 2: "Alive with disease"}

df['Y_60'] = df['Y_60'].map(class_mapping)


age_counts = df.groupby('Y_60')['age'].value_counts(normalize=True).unstack().fillna(0)
percentage_of_ones = age_counts[1] * 100  

# Ensure the percentage_of_ones Series has the correct order
desired_order = ["Alive and well", "Deceased", "Alive with disease"]
percentage_of_ones = percentage_of_ones.reindex(desired_order)

# Perform Chi-square test
contingency_table = pd.crosstab(df['Y_60'], df['age'])
chi2_stat, p_chi2, _, _ = stats.chi2_contingency(contingency_table)

# Assuming there's another feature to analyze with Kruskal-Wallis
# Replace 'Feature' with your actual feature name
kruskal_result = stats.kruskal(
    df[df['Y_60'] == 'Alive and Well']['age'],
    df[df['Y_60'] == 'Deceased']['age'],
    df[df['Y_60'] == 'Alive with disease']['age']
)

# Plotting the percentages of '1's in 'age' for each class
plt.figure(figsize=(10, 6))
sns.barplot(x=percentage_of_ones.index, y=percentage_of_ones.values, order=desired_order)
plt.title(f'Chi-square p-value: {p_chi2:.3e}')
plt.ylabel('Percentage of age > 50 years')
#plt.xlabel('Class')
plt.ylim(0, 100)  # Set y-axis from 0 to 100%


plt.show()




