import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

air=pd.read_excel('C:/Users/Biswa/Desktop/360DigiTMG/12 . Module 12/Assignment/Data Set/EastWestAirlines.xlsx',sheet_name='data')
air=air.drop('ID#',axis=1)

#Normalizong the data
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return(x)

air_norm=norm_func(air)

#Scree plot or Elbow Curve
k=list(range(2,15))
k
TWSS=[]
for i in k: 
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(air_norm)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(air_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,air_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-')
plt.xlabel("No of clusters")
plt.ylabel("Total within ss")
plt.xticks(k)
plt.title('Elbow Curve')  
plt.show()    

model=KMeans(n_clusters=6)
model.fit(air_norm)
model.labels_
md=pd.Series(model.labels_)
air['Clust']=md
air_norm

cm=air.iloc[:,0:].groupby(air.Clust).mean()

