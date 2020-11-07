import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

insu=pd.read_csv('Insurance Dataset.csv')

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

insu_norm=norm_func(insu)

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(insu_norm,method='complete',metric='euclidean')

plt.figure(figsize=(15,8))
plt.xlabel("Index")
plt.ylabel("Distance")
plt.title('Dendogram')

sch.dendrogram(
        z,
        leaf_rotation=0,
        leaf_font_size=8)

from sklearn.cluster import AgglomerativeClustering

insu_c=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(insu_norm)
cluster_labels=pd.Series(insu_c.labels_)
insu['Clust']=cluster_labels
