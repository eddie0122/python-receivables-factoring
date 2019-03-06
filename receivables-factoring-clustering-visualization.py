import pandas as pd
import numpy as np
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, NMF
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

### Read output of factoring
dataFactoring2 = pd.read_csv('./output/data-factoring.csv', low_memory=False)
dataFactoring2.head()

### Normalize : A to I by Revenue (The sum of A to I will be 1)
dataFactoring2Temp = dataFactoring2.iloc[:, 2:]
for col in dataFactoring2Temp.columns[1:]:
    dataFactoring2Temp[col] = dataFactoring2Temp[col] / dataFactoring2Temp['rev']
del col

### Array of A to I
dataFactoring2Temp = dataFactoring2Temp.iloc[:, 1:]
dataNormal = dataFactoring2Temp[dataFactoring2Temp['A'] == 1].reset_index(drop=True) # Properly Retrieved receivable Records
dataOverdue = dataFactoring2Temp[dataFactoring2Temp['A'] < 1].reset_index(drop=True) # Overdue Retrieved receivable Records

### Boxplot of dataOverdue data
plt.figure(figsize=(12,6))
plt.boxplot(([dataOverdue[col] for col in dataOverdue.columns.tolist()]))
plt.xticks(range(0, 10), (list(' ABCDEFGHI')))
plt.show()
 # Columns C - H shows that most of records have alomost 0. Therefore, Making new feature is about the sum of C - H.

### There are 4 features (A, B, C, I)
dataOverdue['C'] = dataOverdue[dataOverdue.columns[2:-1]].sum(axis=1).copy()
dataOverdue.drop(dataOverdue.columns[3:-1], axis=1, inplace=True)


### K-Means : Find Best K value
inertias = []
for k in range(1,5):
    modelKmeans = KMeans(n_clusters=k, max_iter=500, n_jobs=-1)
    modelKmeans.fit(dataOverdue)
    inertias.append(modelKmeans.inertia_)
del k

### Elbow : inertias by Ks values
plt.figure()
plt.plot(inertias, marker='s')
plt.show() # k == 3

### Labeling Cluster with K-Means(3)
modelKmeans = KMeans(n_clusters=3, max_iter=500, n_jobs=-1)
modelKmeans.fit(dataOverdue)
pred = modelKmeans.predict(dataOverdue)
dataOverdue['Segment'] = pred


### PCA : EigenValues and Ratio
modelPCA = PCA(n_components=4)
modelPCA.fit(dataOverdue.iloc[:, :-1].values)
ratioPCA = modelPCA.explained_variance_ratio_.cumsum()

plt.plot(ratioPCA, marker='s')
plt.rc('font', size=8)
for i in range(4):
    plt.text(i, ratioPCA[i]*1.01, '{:.2f}'.format(ratioPCA[i]))
plt.title('Explained Variance Ratio : Cumulative Sum')
plt.show() # 3 dimensional compression

### PCA : 3 Dimensional
modelPCA = PCA(n_components=3)
modelPCA.fit(dataOverdue.iloc[:, :-1].values)
dataPCA = modelPCA.transform(dataOverdue.iloc[:, :-1].values)
dataPCA = DataFrame(dataPCA, columns=['e1', 'e2', 'e3'])
dataPCA['Segment'] = pred

### 3D Plot with PCA and K-Means Clustering
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['blue', 'red', 'green']
for i in range(3):
    temp = dataPCA[dataPCA['Segment'] == i]
    ax.scatter(temp['e1'], temp['e2'], temp['e3'], color=colors[i], label=str(i))
ax.set_title('3 Dimensional Plot with PCA and K-Means')
ax.legend(loc='best')
plt.show()


### TSNE : 2 Dimensional
modelTSNE = TSNE(n_components=2)
dataTSNE = modelTSNE.fit_transform(dataOverdue.iloc[:, :-1].values)
dataTSNE = DataFrame(dataTSNE, columns=['e1', 'e2'])
dataTSNE['Segment'] = pred

sns.scatterplot(data=dataTSNE, x='e1', y='e2', hue='Segment')
plt.title('t-SNE (n_component=2)')
plt.show()

### TSNE : 3 Dimensional
modelTSNE = TSNE(n_components=3)
dataTSNE1 = modelTSNE.fit_transform(dataOverdue.iloc[:, :-1].values)
dataTSNE1 = DataFrame(dataTSNE1, columns=['e1', 'e2', 'e3'])
dataTSNE1['Segment'] = pred

### 3D Plot with t-SNE and K-Means Clustering
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['blue', 'red', 'green']
for i in range(3):
    temp = dataTSNE1[dataTSNE1['Segment'] == i]
    ax.scatter(temp['e1'], temp['e2'], temp['e3'], color=colors[i], label=str(i))
ax.set_title('3 Dimensional Plot with t-SNE and K-Means')
ax.legend(loc='best')
plt.show()