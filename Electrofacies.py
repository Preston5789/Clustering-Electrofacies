# -*- coding: utf-8 -*-
"""
Electrofacies is a model to calculate numerical facies from log data.
The sckit-learn package is used for standardization and clustering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

n_components = 0.85
curve_name = 'FACIES'
n_clusters = 5

df = pd.DataFrame()
xls_file = pd.ExcelFile("C:/Users/Preston Phillips/Documents/CMR.xlsx") # Path to your log data
df = xls_file.parse('Sheet1')
df['DIFF'] = df['SGR']-df['CGR']


curves = ['BFV','FFI', 'SGR', 'CMRP_3MS']
#get boolian array of non-null values
not_null_rows = pd.notnull(df[curves]).any(axis = 1)

#standardizes all data so mean is 0 and std dev is 1
X = StandardScaler().fit_transform(df.loc[not_null_rows, curves])

#PCA fit creates
pc = PCA(n_components = n_components).fit(X)
#pc.transform:reduces number of columns
#pc.transform applies the mapping (transform) to the pc fit. This is where we could test it on other data
components = pd.DataFrame(data = pc.transform(X), index = df[not_null_rows].index)

#Convert to matrix from minibatch
minibatch_input = components.as_matrix()

##Creates new column and assigns it to MiniBatch Cluster
df.loc[not_null_rows, curve_name] = \
                MiniBatchKMeans(n_clusters = n_clusters,
                batch_size = 100).fit_predict(minibatch_input)

#Add 1 to every variable so it goes from 1 to 3
df.loc[not_null_rows, curve_name] += 1

df.to_excel("C:/Users/Preston Phillips/Documents/Spotfire_Katmai/CMR.xlsx")


#print(df )
X_pca = pc.transform(X)
print("original shape:   ", X.shape)
print("transformed shape:", X_pca.shape)
X_new = pc.inverse_transform(X_pca)
fig = plt.figure()
groups = df.groupby('FACIES')
plt.subplot(2, 2, 1)
plt.plot(df.DEPTH, df.TCMR, marker='o', linestyle='', ms=5)
for name, group in groups:
    plt.subplot(2, 2, 1)
    plt.plot(group.DEPTH, group.FFI, marker='o', linestyle='', ms=5, label=name)

    plt.subplot(2, 2, 2)
    plt.plot(group.DEPTH, group.CMRP_3MS, marker='o', linestyle='', ms=5, label=name)



plt.legend()
plt.show()
