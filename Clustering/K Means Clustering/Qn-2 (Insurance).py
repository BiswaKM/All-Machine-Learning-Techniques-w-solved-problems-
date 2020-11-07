import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

insu=pd.read_csv('C:/Users/Biswa/Desktop/360DigiTMG/12 . Module 12/Assignment/Data Set/Insurance Dataset.csv')

def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

insu_norm=norm_func(insu)

k=list(range(2,10))
k

TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(insu_norm)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(insu_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,insu_norm.shape[1]),'euclidean')))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-')
plt.xlabel("No of clusters")
plt.ylabel("Total within ss")
plt.xticks(k)
plt.title('Elbow Curve')  
plt.show()    


model=KMeans(n_clusters=4)
model.fit(insu_norm)
model.labels_
md=pd.Series(model.labels_)
insu['Clust']=md
cm=insu.iloc[:,0:].groupby(insu.Clust).mean()
