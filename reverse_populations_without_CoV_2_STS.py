# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 18:19:59 2024

@author: symeo
"""

import numpy as np
import pandas as pd
import random
from scipy import stats
import math



dfA0 = pd.read_csv(".../arm_A_0d_60d.csv",header=0)
dfA5 = pd.read_csv(".../arm_A_5d_60d.csv",header=0)

dfA0 = dfA0[['age', 'gender', 'karn', 'corm', 'con2Abs', 'IFNg', 'IL10','IL2','IL6','TNFa',
              'CD3','CD4', 'CD56','CD8', 'CPK', 'CRP', 'DD',
                'Ferrit', 'LDH', 'WBC']] 

targetA=dfA5.iloc[:,-1]

dfA5 = dfA5[['age', 'gender', 'karn', 'corm', 'con2Abs', 'IFNg', 'IL10','IL2','IL6','TNFa',
              'CD3','CD4', 'CD56','CD8', 'CPK', 'CRP', 'DD',
                'Ferrit', 'LDH', 'WBC']] 

mean_valuesArmA = np.mean(dfA5, axis=0)
mean_valuesArmA =np.transpose(mean_valuesArmA)


# Calculate the median of each column
median_valuesArmA = np.median(dfA5, axis=0)

# Calculate the standard deviation of each column
stdev_valuesArmA = np.std(dfA5, axis=0)


All5_sc = (dfA5 - mean_valuesArmA) / stdev_valuesArmA





LDA1_60=[-0.10609	,	-0.022235	,	-0.403598	,	-0.00107967	,	0.0715591	,	-0.15622	,	-0.257235	,	0.0414835	,	0.212413	,	0.189775	,	0.0916713	,	0.339126	,	0.2865	,	-0.188976	,	0.157995	,	-0.544067	,	-0.228996	,	-0.168444	,	-0.0468974	,	0.0402487];
LDA2_60=[0.196468	,	0.439385	,	0.130798	,	0.0522937	,	-0.0866093	,	0.17027	,	-0.0617759	,	0.296598	,	0.0143541	,	-0.311235	,	-0.0523495	,	-0.150946	,	-0.459591	,	0.170625	,	-0.190472	,	-0.335992	,	-0.100203	,	-0.270418	,	-0.0020779	,	0.157933];



XX = np.dot(All5_sc, LDA1_60)
YY = np.dot(All5_sc, LDA2_60)

combined_LDA_armA = np.vstack((XX, YY, targetA)).T  # Transpose to get columns X, Y, Z

sorted_array_ArmA = combined_LDA_armA[combined_LDA_armA[:, 2].argsort()]

HealthyArmA=combined_LDA_armA[combined_LDA_armA[:, 2]==0]
SuccumedArmA=combined_LDA_armA[combined_LDA_armA[:, 2]==1]
IllArmA=combined_LDA_armA[combined_LDA_armA[:, 2]==2]


aveHealthyArmA=np.mean(HealthyArmA,axis=0)
aveSuccumedArmA=np.mean(SuccumedArmA,axis=0)
aveIllArmA=np.mean(IllArmA,axis=0)

aveHealthyArmA=aveHealthyArmA[:-1,]
aveSuccumedArmA=aveSuccumedArmA[:-1,]
aveIllArmA=aveIllArmA[:-1,]

cov_matrix_ArmA_Healthy = np.cov(HealthyArmA[:,:-1], rowvar=False)
cov_matrix_ArmA_Succumed = np.cov(SuccumedArmA[:,:-1], rowvar=False)
cov_matrix_ArmA_Ill = np.cov(IllArmA[:,:-1], rowvar=False)

lenghtHealthy=HealthyArmA.shape[0]
lenghtSuccumed=SuccumedArmA.shape[0]
lenghtIll=IllArmA.shape[0]

S=((lenghtHealthy-1)*cov_matrix_ArmA_Healthy+(lenghtSuccumed-1)*cov_matrix_ArmA_Succumed+(lenghtIll-1)*cov_matrix_ArmA_Ill)/((lenghtHealthy-1)+(lenghtSuccumed-1)+(lenghtIll-1))

Sinv = np.linalg.inv(S)

d0_Healthy=-0.5*np.dot(np.dot(aveHealthyArmA,Sinv),aveHealthyArmA.T)
di_Healthy=np.dot(aveHealthyArmA,Sinv)

d0_Succumed=-0.5*np.dot(np.dot(aveSuccumedArmA,Sinv),aveSuccumedArmA.T)
di_Succumed=np.dot(aveSuccumedArmA,Sinv)


d0_Ill=-0.5*np.dot(np.dot(aveIllArmA,Sinv),aveIllArmA.T)
di_Ill=np.dot(aveIllArmA,Sinv)

paramArmA = np.array([
    np.concatenate([[d0_Healthy], di_Healthy, [np.log10(lenghtHealthy / 57)]]),
    np.concatenate([[d0_Succumed], di_Succumed, [np.log10(lenghtSuccumed / 57)]]),
    np.concatenate([[d0_Ill], di_Ill, [np.log10(lenghtIll / 57)]])
])


dfA0=dfA0.iloc[:,4:]
dfA5=dfA5.iloc[:,4:]


dfrA=(dfA5-dfA0)/5

median_valuesderArmA = np.median(dfrA, axis=0)
iqr_valuesArmA = dfrA.quantile(0.75) - dfrA.quantile(0.25)
BmaxarmA=median_valuesderArmA+iqr_valuesArmA/2









dfB0 = pd.read_csv(".../arm_B_0d_60d.csv")
dfB5 = pd.read_csv(".../arm_B_5d_60d.csv")
targetB=dfB5.iloc[:,-1]
dfB0 = dfB0[['age', 'gender', 'karn', 'corm', 'con2Abs', 'IFNg', 'IL10','IL2','IL6','TNFa',
              'CD3','CD4', 'CD56','CD8', 'CPK', 'CRP', 'DD',
                'Ferrit', 'LDH', 'WBC']] 

dfB5 = dfB5[['age', 'gender', 'karn', 'corm', 'con2Abs', 'IFNg', 'IL10','IL2','IL6','TNFa',
              'CD3','CD4', 'CD56','CD8', 'CPK', 'CRP', 'DD',
                'Ferrit', 'LDH', 'WBC']] 

mean_valuesArmB = np.mean(dfB5, axis=0)
mean_valuesArmB =np.transpose(mean_valuesArmB)
median_valuesArmB = np.median(dfB5, axis=0)
stdev_valuesArmB = np.std(dfB5, axis=0)

Bll5_sc = (dfB5 - mean_valuesArmB) / stdev_valuesArmB



LDB1_60=[0.261303	,	0.191785	,	0.0165054	,	-0.0717917	,	0.0592559	,	0.574401	,	-0.224677	,	-0.17277	,	-0.183074	,	-0.175954	,	-0.435006	,	0.00725546	,	0.347487	,	-0.0672569	,	0.117825	,	0.0261213	,	-0.0512724	,	-0.256619	,	0.0754037	,	0.0892868];
LDB2_60=[0.00423973	,	0.0536658	,	-0.156888	,	-0.0122554	,	0.0271917	,	0.682014	,	-0.594037	,	-0.00290295	,	-0.0141075	,	0.0517024	,	-0.142667	,	0.0730172	,	0.0429811	,	-0.0852778	,	-0.089634	,	-0.205841	,	-0.107569	,	-0.137659	,	0.17447	,	0.0672562];



XXB = np.dot(Bll5_sc, LDB1_60)
YYB = np.dot(Bll5_sc, LDB2_60)

combined_LDA_armB = np.vstack((XXB, YYB, targetB)).T  # Transpose to get columns X, Y, Z

sorted_array_ArmB = combined_LDA_armB[combined_LDA_armB[:, 2].argsort()]

HealthyArmB=combined_LDA_armB[combined_LDA_armB[:, 2]==0]
SuccumedArmB=combined_LDA_armB[combined_LDA_armB[:, 2]==1]
IllArmB=combined_LDA_armB[combined_LDA_armB[:, 2]==2]


aveHealthyArmB=np.mean(HealthyArmB,axis=0)
aveSuccumedArmB=np.mean(SuccumedArmB,axis=0)
aveIllArmB=np.mean(IllArmB,axis=0)

aveHealthyArmB=aveHealthyArmB[:-1,]
aveSuccumedArmB=aveSuccumedArmB[:-1,]
aveIllArmB=aveIllArmB[:-1,]

cov_matrix_ArmB_Healthy = np.cov(HealthyArmB[:,:-1], rowvar=False)
cov_matrix_ArmB_Succumed = np.cov(SuccumedArmB[:,:-1], rowvar=False)
cov_matrix_ArmB_Ill = np.cov(IllArmB[:,:-1], rowvar=False)

lenghtHealthyB=HealthyArmB.shape[0]
lenghtSuccumedB=SuccumedArmB.shape[0]
lenghtIllB=IllArmB.shape[0]

SB=((lenghtHealthyB-1)*cov_matrix_ArmB_Healthy+(lenghtSuccumedB-1)*cov_matrix_ArmB_Succumed+(lenghtIllB-1)*cov_matrix_ArmB_Ill)/((lenghtHealthyB-1)+(lenghtSuccumedB-1)+(lenghtIllB-1))

SinvB = np.linalg.inv(SB)

d0_HealthyB=-0.5*np.dot(np.dot(aveHealthyArmB,SinvB),aveHealthyArmB.T)
di_HealthyB=np.dot(aveHealthyArmB,SinvB)

d0_SuccumedB=-0.5*np.dot(np.dot(aveSuccumedArmB,SinvB),aveSuccumedArmB.T)
di_SuccumedB=np.dot(aveSuccumedArmB,SinvB)


d0_IllB=-0.5*np.dot(np.dot(aveIllArmB,SinvB),aveIllArmB.T)
di_IllB=np.dot(aveIllArmB,SinvB)

paramArmB = np.array([
    np.concatenate([[d0_HealthyB], di_HealthyB, [np.log10(lenghtHealthyB / 30)]]),
    np.concatenate([[d0_SuccumedB], di_SuccumedB, [np.log10(lenghtSuccumedB / 30)]]),
    np.concatenate([[d0_IllB], di_IllB, [np.log10(lenghtIllB / 30)]])
])



dfB0=dfB0.iloc[:,4:]
dfB5=dfB5.iloc[:,4:]
dfrB=(dfB5-dfB0)/5


median_valuesderArmB = np.median(dfrB, axis=0)
iqr_valuesArmB = dfrB.quantile(0.75) - dfrB.quantile(0.25)
BmaxarmB=median_valuesderArmB+iqr_valuesArmB/2



# Set the number of times to generate random numbers
num_samples = 1000

# Initialize a list to store the random numbers
random_numbersA = []
Healthy_valuesA = []
Succumed_valuesA = []
number_aliveA=[]

for _ in range(num_samples):
    
    A0= pd.read_csv(".../arm_A_0d_60d.csv")
    A0 = A0[['age', 'gender', 'karn', 'corm', 'con2Abs', 'IFNg', 'IL10','IL2','IL6','TNFa',
                  'CD3','CD4', 'CD56','CD8', 'CPK', 'CRP', 'DD',
                    'Ferrit', 'LDH', 'WBC']] 
    A0=np.array(A0)

    pr_healthy_s = np.zeros(57)
    pr_deceased_s = np.zeros(57)
    pr_ill_s = np.zeros(57)

    day = 5

    
     # Bmax vector: the maximum rate of change for each biomarker after 5 days (median_rate_of_change + 1/2*IQR_rate_of_change)
    Bmax = BmaxarmB
    
    # A5_con: exclude the constant information [age, gender Karnofski, corm]
    import numpy as np
    A5_con = A0[:, :4]
    Arest=A0[:,4:]

    # Initialize A5rest as an array of zeros with the same shape as A5_con
    A5rest = np.zeros_like(Arest)
    
    # Generate random transitions for each patient
    for j in range(57):
        for i in range(16):
            mean_value =  0
            std_dev = abs(5  * (Bmax[i]))
            

            A5rest[j, i] =A0[j, i + 4] + np.random.normal(mean_value, std_dev)
            if A5rest[j, i] < 0 :
                A5rest[j, i]=0
                
    # Concatenate baseline characteristics with new values biomarkers
    Aall5 = np.concatenate((A5_con, A5rest), axis=1)
    

    mean_sc = mean_valuesArmB
     

    std_sc=stdev_valuesArmB
    
    All5_sc = np.zeros((57, 20))  # Initialize All5_sc as an array of zeros with the same shape
    
    for j in range(57):
        for i in range(20):
            All5_sc[j, i] = (Aall5[j, i] - mean_sc[i]) / std_sc[i]
    
    LDA1_60=[0.261303	,	0.191785	,	0.0165054	,	-0.0717917	,	0.0592559	,	0.574401	,	-0.224677	,	-0.17277	,	-0.183074	,	-0.175954	,	-0.435006	,	0.00725546	,	0.347487	,	-0.0672569	,	0.117825	,	0.0261213	,	-0.0512724	,	-0.256619	,	0.0754037	,	0.0892868];
    LDA2_60=[0.00423973	,	0.0536658	,	-0.156888	,	-0.0122554	,	0.0271917	,	0.682014	,	-0.594037	,	-0.00290295	,	-0.0141075	,	0.0517024	,	-0.142667	,	0.0730172	,	0.0429811	,	-0.0852778	,	-0.089634	,	-0.205841	,	-0.107569	,	-0.137659	,	0.17447	,	0.0672562];

    
    
    
    XX = np.dot(All5_sc, LDA1_60)
    YY = np.dot(All5_sc, LDA2_60)
    
    
    
    param =paramArmB 
    
    

    
    # Initialize arrays for results
    healthy = np.zeros(57)
    deceased = np.zeros(57)
    ill = np.zeros(57)
    pr_healthy = np.zeros(57)
    pr_deceased = np.zeros(57)
    pr_ill = np.zeros(57)
    
    for i in range(57):
        healthy[i] = param[0, 0] + param[0, 1] * XX[i] + param[0, 2] * YY[i] + param[0, 3]
        deceased[i] = param[1, 0] + param[1, 1] * XX[i] + param[1, 2] * YY[i] + param[1, 3]
        ill[i] = param[2, 0] + param[2, 1] * XX[i] + param[2, 2] * YY[i] + param[2, 3]
    
        exp_healthy = np.exp(healthy[i])
        exp_deceased = np.exp(deceased[i])
        exp_ill = np.exp(ill[i])
    
        denominator = exp_healthy + exp_deceased + exp_ill
    
        pr_healthy[i] = exp_healthy / denominator
        pr_deceased[i] = exp_deceased / denominator
        pr_ill[i] = exp_ill / denominator
    
    
    Prob = np.column_stack((pr_healthy, pr_deceased, pr_ill))
    
    for i in range(57):
        if np.isnan(Prob[i, 1]):
            Prob[i, 1] = 1
            Prob[i, 0] = 0
            Prob[i, 2] = 0
        else:
            Prob[i, 0] = Prob[i, 0]-pr_deceased_s[i]
            Prob[i, 1] = Prob[i, 1]+pr_deceased_s[i]
            Prob[i, 2] = Prob[i, 2]-pr_deceased_s[i]
    
    index_healthy = np.zeros(57, dtype=int)
    index_ill = np.zeros(57, dtype=int)
    index_deceased = np.zeros(57, dtype=int)
    
    for i in range(57):
        if Prob[i, 0] > Prob[i, 1] and Prob[i, 0] > Prob[i, 2]:
            index_healthy[i] = 1
    
    for i in range(57):
        if Prob[i, 2] > Prob[i, 1] and Prob[i, 2] > Prob[i, 0]:
            index_ill[i] = 1
    
    for i in range(57):
        if Prob[i, 1] > Prob[i, 0] and Prob[i, 1] > Prob[i, 2]:
            index_deceased[i] = 1
    
    # Calculate the percentage of alive patients
    alive_count = np.sum(index_healthy) + np.sum(index_ill)
    perc_healthy = alive_count / 57
    
    
    random_numbersA.append(perc_healthy)
    perc_succum=1-perc_healthy
    
    if np.round(alive_count)>alive_count:
        alive_count=alive_count+1
    else:
        alive_count=alive_count
    
    number_aliveA.append((alive_count-43)/43)
    
    Healthy_valuesA.append(perc_healthy)
    Succumed_valuesA.append(perc_succum)


averageA = sum(random_numbersA) / num_samples

# Calculate the squared differences from the mean
squared_diff = [(x - averageA)**2 for x in random_numbersA]

# Calculate the variance
varianceA = sum(squared_diff) / (num_samples - 1)  # Use (n-1) for sample standard deviation

# Calculate the standard deviation
std_devA = math.sqrt(varianceA)

statisticA, p_valueA = stats.mannwhitneyu(Healthy_valuesA, Succumed_valuesA)

# Output the results
print("Mann-Whitney U statistic:", statisticA)
print("P-value:", p_valueA)

# Interpret the results
alpha = 0.05  # Set your significance level
if p_valueA < alpha:
    print("There is a significant difference between the two lists.")
else:
    print("There is no significant difference between the two lists.")

#===================================================================================================


# Initialize a list to store the random numbers
random_numbersB = []
Healthy_valuesB = []
Succumed_valuesB = []
number_aliveB=[]


for _ in range(num_samples):

    A0= pd.read_csv(".../arm_B_0d_60d.csv")
    A0 = A0[['age', 'gender', 'karn', 'corm', 'con2Abs', 'IFNg', 'IL10','IL2','IL6','TNFa',
                  'CD3','CD4', 'CD56','CD8', 'CPK', 'CRP', 'DD',
                    'Ferrit', 'LDH', 'WBC']] 
    A0=np.array(A0)
    

    pr_healthy_s = np.zeros(57)
    pr_deceased_s = np.zeros(57)
    pr_ill_s = np.zeros(57)

    day = 5

    
    # Bmax vector: the maximum rate of change for each biomarker after 5 days (median_rate_of_change + 1/2*IQR_rate_of_change)
    Bmax = BmaxarmA
    
    # A5_con: exclude the constant information [age, gender Karnofski, corm]
    
    A5_con = A0[:, :4]
    Arest=A0[:,4:]

    # Initialize A5rest as an array of zeros with the same shape as A5_con
    A5rest = np.zeros_like(Arest)
    
    # Generate random transitions for each patient
    for j in range(30):
        for i in range(16):
            mean_value = 0
            std_dev = abs(5  * (Bmax[i]))
            

            A5rest[j, i] =A0[j, i + 4] + np.random.normal(mean_value, std_dev)
            if A5rest[j, i] < 0 :
                A5rest[j, i]=0
                
    # Concatenate baseline characteristics with new values biomarkers
    Aall5 = np.concatenate((A5_con, A5rest), axis=1)
    
   
    mean_sc =mean_valuesArmA
   
    std_sc =stdev_valuesArmA
    
    All5_sc = np.zeros((30, 20))  # Initialize All5_sc as an array of zeros with the same shape
    
    for j in range(30):
        for i in range(20):
            All5_sc[j, i] = (Aall5[j, i] - mean_sc[i]) / std_sc[i]
    
    LDA1_60=[-0.10609	,	-0.022235	,	-0.403598	,	-0.00107967	,	0.0715591	,	-0.15622	,	-0.257235	,	0.0414835	,	0.212413	,	0.189775	,	0.0916713	,	0.339126	,	0.2865	,	-0.188976	,	0.157995	,	-0.544067	,	-0.228996	,	-0.168444	,	-0.0468974	,	0.0402487];
    LDA2_60=[0.196468	,	0.439385	,	0.130798	,	0.0522937	,	-0.0866093	,	0.17027	,	-0.0617759	,	0.296598	,	0.0143541	,	-0.311235	,	-0.0523495	,	-0.150946	,	-0.459591	,	0.170625	,	-0.190472	,	-0.335992	,	-0.100203	,	-0.270418	,	-0.0020779	,	0.157933];

    
    
    
    XX = np.dot(All5_sc, LDA1_60)
    YY = np.dot(All5_sc, LDA2_60)
    
    
    
    param = paramArmA
    
    # Initialize arrays for results
    healthy = np.zeros(30)
    deceased = np.zeros(30)
    ill = np.zeros(30)
    pr_healthy = np.zeros(30)
    pr_deceased = np.zeros(30)
    pr_ill = np.zeros(30)
    
    for i in range(30):
        healthy[i] = param[0, 0] + param[0, 1] * XX[i] + param[0, 2] * YY[i] + param[0, 3]
        deceased[i] = param[1, 0] + param[1, 1] * XX[i] + param[1, 2] * YY[i] + param[1, 3]
        ill[i] = param[2, 0] + param[2, 1] * XX[i] + param[2, 2] * YY[i] + param[2, 3]
    
        exp_healthy = np.exp(healthy[i])
        exp_deceased = np.exp(deceased[i])
        exp_ill = np.exp(ill[i])
    
        denominator = exp_healthy + exp_deceased + exp_ill
    
        pr_healthy[i] = exp_healthy / denominator
        pr_deceased[i] = exp_deceased / denominator
        pr_ill[i] = exp_ill / denominator
    
    
    Prob = np.column_stack((pr_healthy, pr_deceased, pr_ill))
    
    
    index_healthy = np.zeros(30, dtype=int)
    index_ill = np.zeros(30, dtype=int)
    index_deceased = np.zeros(30, dtype=int)
    
    for i in range(30):
        if Prob[i, 0] > Prob[i, 1] and Prob[i, 0] > Prob[i, 2]:
            index_healthy[i] = 1
    
    for i in range(30):
        if Prob[i, 2] > Prob[i, 1] and Prob[i, 2] > Prob[i, 0]:
            index_ill[i] = 1
    
    for i in range(30):
        if Prob[i, 1] > Prob[i, 0] and Prob[i, 1] > Prob[i, 2]:
            index_deceased[i] = 1
    
    # Calculate the percentage of alive patients
    alive_countB = np.sum(index_healthy) + np.sum(index_ill)
    perc_healthyB = alive_countB / 30
    perc_succum=1-perc_healthyB
    
    random_numbersB.append(perc_healthyB)
    Healthy_valuesB.append(perc_healthyB)
    Succumed_valuesB.append(perc_succum)

    if np.round(alive_countB)>alive_countB:
        alive_countB=alive_countB+1
    else:
        alive_countB=alive_countB
    
    number_aliveB.append((alive_countB-15)/15)
    
    Healthy_valuesB.append(perc_healthy)
    Succumed_valuesB.append(perc_succum)


averageB = sum(random_numbersB) / num_samples



# Calculate the squared differences from the mean
squared_diff = [(x - averageB)**2 for x in random_numbersB]

# Calculate the variance
variance = sum(squared_diff) / (num_samples - 1)  # Use (n-1) for sample standard deviation

# Calculate the standard deviation
std_dev = math.sqrt(variance)

statisticB, p_valueB = stats.mannwhitneyu(Healthy_valuesB, Succumed_valuesB)

# Output the results
print("Mann-Whitney U statistic:", statisticA)
print("P-value:", p_valueA)

# Interpret the results
alpha = 0.05  # Set your significance level
if p_valueA < alpha:
    print("There is a significant difference between the two lists.")
else:
    print("There is no significant difference between the two lists.")


import matplotlib.pyplot as plt
import numpy as np


# Calculate sums
sum_alive_A = np.mean(number_aliveA)
sum_alive_B = np.mean(number_aliveB)

# Calculate standard deviations
std_dev_A = np.std(number_aliveA)
std_dev_B = np.std(number_aliveB)

# Creating the figure and axis objects
fig, ax = plt.subplots(figsize=(8, 6))

# Data for plotting
categories = ['Arm A given \n SoC only', 'Arm B given \n CoV-2-STs + SoC']
values = [sum_alive_A, sum_alive_B]
errors = [std_dev_A, std_dev_B]  # Standard deviations


# Creating horizontal bar plot with error bars
ax.barh(categories, values, xerr=errors, color=['red', '#6B9AC4'], capsize=5)  # capsize controls the size of error bar caps

# Adding labels
ax.set_xlabel('Percentage change of alive patients')
ax.set_title('Percentage change of alive patients in theoretical \n what-if scenarios')

plt.tight_layout()

plt.savefig('.../filename.png', dpi=300)


# Show the plot
plt.show()


