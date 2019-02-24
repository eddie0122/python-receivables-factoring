import pandas as pd
import numpy as np
from pandas import DataFrame, Series

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

### Read output of factoring
dataFactoring2 = pd.read_csv('./output/data-factoring.csv', low_memory=False)

### Normalize : A to I by Revenue (The sum of A to I will be 1)
dataFactoring2Temp = dataFactoring2.iloc[:, 2:]
for col in dataFactoring2Temp.columns[1:]:
    dataFactoring2Temp[col] = dataFactoring2Temp[col] / dataFactoring2Temp['rev']
del col

### Array of A to I
dataFactoring2Temp = dataFactoring2Temp.iloc[:, 1:].values


### Kmeans : Find Best K value
inertias = []
for k in range(1,10):
    modelKmeans = KMeans(n_clusters=k, max_iter=500)
    modelKmeans.fit(dataFactoring2Temp)
    inertias.append(modelKmeans.inertia_)
del k

### Elbow plot : inertias by Ks values
plt.figure()
plt.plot(inertias, marker='s')
plt.show() # k = 4

### Kmeans
modelKmeans = KMeans(n_clusters=2, max_iter=500)
modelKmeans.fit(dataFactoring2Temp)
modelKmeansPred = modelKmeans.predict(dataFactoring2Temp)

dataFactoring2Kmeans = DataFrame(dataFactoring2Temp, columns=list('ABCDEFGHI'))
dataFactoring2Kmeans['Segment'] = modelKmeansPred

# dataCorr = dataFactoring2Kmeans.drop('Segment', axis=1).corr()
# sns.heatmap(dataCorr, annot=True, fmt='.3f', cmap='BrBG')
# plt.show()

_, temp = train_test_split(dataFactoring2Kmeans, stratify=dataFactoring2Kmeans['Segment'], test_size=.02)
temp.reset_index(drop=True, inplace=True)


modelTSNE = TSNE(n_components=2)
compressionTSNE = modelTSNE.fit_transform(temp.iloc[:, :-1].values)
dataTSNE = pd.concat([temp.iloc[:, -1], DataFrame(compressionTSNE, columns=['A', 'B'])], axis=1)

# temp['bankrupt'] = np.nan
# temp.loc[temp['I'] > 0, 'bankrupt'] = 1
# temp.loc[temp['bankrupt'].isnull(), 'bankrupt'] = 0

# dataTSNE['bankrupt'] = temp['bankrupt']
# dataTSNE.head()

fig, ax = plt.subplots(1,1, figsize=(12,12))
sns.scatterplot(data=dataTSNE, x='A', y='B', hue='Segment', ax=ax)
# sns.scatterplot(data=dataTSNE, x='A', y='B', hue='bankrupt', ax=ax[1])
plt.show()


modelPCA = PCA(n_components=2)
compressPCA = modelPCA.fit_transform(dataFactoring2Kmeans.iloc[:, :-1].values)
dataPCA = pd.concat([dataFactoring2Kmeans['Segment'], DataFrame(compressPCA, columns=['PCA1', 'PCA2'])], axis=1)

sns.scatterplot(data=dataPCA, x='PCA1', y='PCA2', hue='Segment')
plt.show()

