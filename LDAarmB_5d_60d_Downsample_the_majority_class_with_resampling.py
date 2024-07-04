# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 20:55:15 2024

@author: symeo
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
    import pandas as pd
    from sklearn.utils import resample
    import csv
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import f1_score
    import seaborn as sns
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
    
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import precision_score, classification_report
    
    data= pd.read_csv(".../arm_B_5d_60d.csv")
    
    

    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    df = pd.concat([pd.DataFrame(X), pd.Series(y, name='target')], axis=1)


    class_counts = df['target'].value_counts()
    

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    

    minority_class_size = class_counts.min()
    majority_class_size=class_counts.max()
    mid_class=30-minority_class_size-majority_class_size
    

    df_majority = df[df['target'] == majority_class]
    df_minority = df[df['target'] == minority_class]
    df_mid = df[df['target'] == 0]
    
    num_iterations = 19
    random_states = [123, 234, 345, 456, 567, 810, 743, 251,222,358,111,222,333,444,555,666,777,888,999]  # Example random states

    results = []
    ppp = []

    for i in range(num_iterations):
    
        # Downsample the majority class
        df_majority_downsampled = resample(df_majority, 
                                       replace=False,    # sample without replacement
                                       n_samples=mid_class,     # to match minority class
                                       random_state=random_states[i]) # reproducible results
        df_mid_downsampled = resample(df_mid, 
                                       replace=False,    # sample without replacement
                                       n_samples=mid_class,     # to match minority class
                                       random_state=123) # reproducible results
    
        # Combine the downsampled majority class with the original minority class(es)
        df_balanced = pd.concat([df_majority_downsampled, df_minority, df_mid_downsampled])
    
        # Shuffle the dataset to mix up the order of rows
        df_balanced = df_balanced.sample(frac=1, random_state=123).reset_index(drop=True)
    
        # Extract features and target variables for the balanced dataset
        X_balanced = df_balanced.iloc[:, :-1].values
        y_balanced = df_balanced.iloc[:, -1].values
    
    

    
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X=sc.fit_transform(X_balanced)


        y=y_balanced

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


        from matplotlib.font_manager import FontProperties

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
    
    

        from sklearn.metrics import confusion_matrix, accuracy_score
        from sklearn.model_selection import cross_val_score
        lin_score=cross_val_score(classifier,X_projected.real,y,cv=3,scoring='accuracy')
        
        cv = StratifiedKFold(n_splits=3)
        y_pred = cross_val_predict(classifier, X_projected.real, y, cv=cv)
        

        precision_per_class = precision_score(y, y_pred, average=None)
        ppp.append(precision_per_class)
        

        print("Precision per class:")
        for k, precision in enumerate(precision_per_class):
            print(f"Class {k}: {precision:.4f}")
        
 
        macro_precision = precision_score(y, y_pred, average='macro')
        micro_precision = precision_score(y, y_pred, average='micro')
        
        print(f"\nMacro Average Precision: {macro_precision:.4f}")
        print(f"Micro Average Precision: {micro_precision:.4f}") 
        
    
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
        #    plt.colorbar()
    
        from matplotlib.font_manager import FontProperties
        # Add legend
        font = FontProperties(family='Times New Roman', weight='bold')
        font.set_weight('bold')
        
        plt.xlabel("Linear Discriminant 1", fontproperties=font)
        plt.ylabel("Linear Discriminant 2", fontproperties=font)
        plt.title("Scatter Plot of Linear Discriminants", fontproperties=font)
        classes = ['Alive and well', 'Died', 'Alive with disease']
        plt.legend(handles=scatter.legend_elements()[0], title="Classes",labels=classes, prop=font)
        
        plt.show()
        
        
        precision_per_fold = []
       
       

        for train_index, test_index in cv.split(X, y):
           X_train, X_test = X_projected.real[train_index], X_projected.real[test_index]
           y_train, y_test = y[train_index], y[test_index]
       
          
           y_pred = classifier.predict(X_test)
          

           precision_fold = precision_score(y_test, y_pred, average=None)
           precision_per_fold.append(precision_fold)
       

        precision_per_fold = np.array(precision_per_fold)
       
       # Calculate standard deviations for precision per class
        std_dev_per_class = np.std(precision_per_fold, axis=0)
       
       # Print standard deviations for precision per class
        print("Standard Deviations of Precision per class:")
        for k, std_dev in enumerate(std_dev_per_class):
           print(f"Class {k}: {std_dev:.4f}")
       
        
       
      
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
    STD_f1_score = np.std(f1_scores_per_fold)
    print(f"Average F1 Score (Macro): {average_f1_score:.4f}")

    # Store results
    results.append({
        'iteration': i+1,
        'accuracy': np.mean(lin_score),
        'std_accuracy':np.std(lin_score),
        'precision_per_class': ppp,
        'macro_precision': macro_precision,
        'micro_precision': micro_precision,
        'f1_score': average_f1_score,
        'std_f1_score': STD_f1_score
    })