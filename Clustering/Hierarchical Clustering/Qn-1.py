import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

air=pd.read_excel('EastWestAirlines.xlsx',sheet_name='data')

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

air_norm=norm_func(air) 

air_norm=air_norm.drop('ID#',axis=1)
                       
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(air_norm,method='complete',metric='euclidean')                       

plt.figure(figsize=(12,8))                       
plt.title('Hierarchical Clustering Dendogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(
        z,
        leaf_rotation=0,
        leaf_font_size=8)
plt.show()

from sklearn.cluster import AgglomerativeClustering

air_c=AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(air_norm)

cluster_labels=pd.Series(air_c.labels_)
air['Clust']=cluster_labels

c=air.iloc[:,0:].groupby(air.Clust==2)

q=air.loc[air['Clust']==2]
