import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

crime=pd.read_csv('crime_data.csv')

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
crime=crime.drop(['Unnamed: 0'],axis=1)

crime_norm=norm_func(crime)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(crime_norm,method='complete',metric='euclidean')

plt.plot(figuresize=(12,8))
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('Dendrogram')

sch.dendrogram(
        z,
        leaf_rotation=0,
        leaf_font_size=8)
plt.show()

from sklearn.cluster import AgglomerativeClustering

crime_c=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='complete').fit(crime_norm)

cluster_labels=pd.Series(crime_c.labels_)

crime['Clust']=cluster_labels
