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

mean_valuesArmA = np.mean(dfA5, axis=0)
mean_valuesArmA =np.transpose(mean_valuesArmA)


# Calculate the median of each column
median_valuesArmA = np.median(dfA5, axis=0)

# Calculate the standard deviation of each column
stdev_valuesArmA = np.std(dfA5, axis=0)


All5_sc = (dfA5 - mean_valuesArmA) / stdev_valuesArmA

All5_sc=All5_sc.iloc[:,:-1]
targetA=dfA5.iloc[:,-1]


LDA1_60=[-0.108683	,	-0.0420155	,	-0.393404	,	-0.00771316	,	0.0618027	,	-0.153663	,	-0.244385	,	0.037569	,	0.224888	,	0.172452	,	0.0373457	,	0.364563	,	0.303837	,	-0.212638	,	0.117837	,	0.163322	,	-0.523906	,	-0.219322	,	-0.160377	,	-0.0351561	,	0.0280404];
LDA2_60=[0.191084	,	0.442174	,	0.113815	,	0.0568398	,	-0.0733973	,	0.160557	,	-0.0733802	,	0.289489	,	0.00262462	,	-0.281531	,	0.00348248	,	-0.168552	,	-0.457881	,	0.186505	,	-0.115825	,	-0.187739	,	-0.345909	,	-0.107096	,	-0.269018	,	-0.0137775	,	0.16419];



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
dfA5=dfA5.iloc[:,4:-1]


dfrA=(dfA5-dfA0)/5

median_valuesderArmA = np.median(dfrA, axis=0)
iqr_valuesArmA = dfrA.quantile(0.75) - dfrA.quantile(0.25)
BmaxarmA=median_valuesderArmA+iqr_valuesArmA/2









dfB0 = pd.read_csv(".../arm_B_0d_60d.csv")
dfB5 = pd.read_csv(".../arm_B_5d_60d.csv")


mean_valuesArmB = np.mean(dfB5, axis=0)
mean_valuesArmB =np.transpose(mean_valuesArmB)
median_valuesArmB = np.median(dfB5, axis=0)
stdev_valuesArmB = np.std(dfB5, axis=0)

Bll5_sc = (dfB5 - mean_valuesArmB) / stdev_valuesArmB

Bll5_sc=Bll5_sc.iloc[:,:-1]
targetB=dfB5.iloc[:,-1]


LDB1_60=[0.122859	,	0.107848	,	-0.0333922	,	-0.0383402	,	0.0582159	,	0.714879	,	-0.426319	,	-0.0702995	,	-0.180346	,	-0.0750462	,	-0.293085	,	0.0831691	,	0.19736	,	-0.154526	,	0.110154	,	0.0384089	,	0.0263093	,	-0.162759	,	-0.158729	,	0.095775	,	0.0244332];
LDB2_60=[-0.0344733	,	0.0150145	,	-0.111225	,	-0.000208837	,	0.0381113	,	0.708166	,	-0.576096	,	0.035663	,	-0.0914262	,	0.0532741	,	-0.0993307	,	0.121252	,	0.00863928	,	-0.168617	,	0.130049	,	-0.0708226	,	-0.0662404	,	-0.197994	,	-0.0624517	,	0.127921	,	-0.00717849];



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
dfB5=dfB5.iloc[:,4:-1]
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
        for i in range(17):
            mean_value =  0
            std_dev = abs(5  * (Bmax[i]))
            

            A5rest[j, i] =A0[j, i + 4] + np.random.normal(mean_value, std_dev)
            if A5rest[j, i] < 0 :
                A5rest[j, i]=0
                
    # Concatenate baseline characteristics with new values biomarkers
    Aall5 = np.concatenate((A5_con, A5rest), axis=1)
    

    mean_sc = mean_valuesArmB
     

    std_sc=stdev_valuesArmB
    
    All5_sc = np.zeros((57, 21))  # Initialize All5_sc as an array of zeros with the same shape
    
    for j in range(57):
        for i in range(21):
            All5_sc[j, i] = (Aall5[j, i] - mean_sc[i]) / std_sc[i]
    
    
    
    LDA1_60=[0.122859	,	0.107848	,	-0.0333922	,	-0.0383402	,	0.0582159	,	0.714879	,	-0.426319	,	-0.0702995	,	-0.180346	,	-0.0750462	,	-0.293085	,	0.0831691	,	0.19736	,	-0.154526	,	0.110154	,	0.0384089	,	0.0263093	,	-0.162759	,	-0.158729	,	0.095775	,	0.0244332];
    LDA2_60=[-0.0344733	,	0.0150145	,	-0.111225	,	-0.000208837	,	0.0381113	,	0.708166	,	-0.576096	,	0.035663	,	-0.0914262	,	0.0532741	,	-0.0993307	,	0.121252	,	0.00863928	,	-0.168617	,	0.130049	,	-0.0708226	,	-0.0662404	,	-0.197994	,	-0.0624517	,	0.127921	,	-0.00717849];

    
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
        for i in range(17):
            mean_value = 0
            std_dev = abs(5  * (Bmax[i]))
            

            A5rest[j, i] =A0[j, i + 4] + np.random.normal(mean_value, std_dev)
            if A5rest[j, i] < 0 :
                A5rest[j, i]=0
                
    # Concatenate baseline characteristics with new values biomarkers
    Aall5 = np.concatenate((A5_con, A5rest), axis=1)
    
   
    mean_sc =mean_valuesArmA
   
    std_sc =stdev_valuesArmA
    
    All5_sc = np.zeros((30, 21))  # Initialize All5_sc as an array of zeros with the same shape
    
    for j in range(30):
        for i in range(21):
            All5_sc[j, i] = (Aall5[j, i] - mean_sc[i]) / std_sc[i]
   
    
    LDA1_60=[-0.108683	,	-0.0420155	,	-0.393404	,	-0.00771316	,	0.0618027	,	-0.153663	,	-0.244385	,	0.037569	,	0.224888	,	0.172452	,	0.0373457	,	0.364563	,	0.303837	,	-0.212638	,	0.117837	,	0.163322	,	-0.523906	,	-0.219322	,	-0.160377	,	-0.0351561	,	0.0280404];
    LDA2_60=[0.191084	,	0.442174	,	0.113815	,	0.0568398	,	-0.0733973	,	0.160557	,	-0.0733802	,	0.289489	,	0.00262462	,	-0.281531	,	0.00348248	,	-0.168552	,	-0.457881	,	0.186505	,	-0.115825	,	-0.187739	,	-0.345909	,	-0.107096	,	-0.269018	,	-0.0137775	,	1.64E-01];
    
    
    
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


