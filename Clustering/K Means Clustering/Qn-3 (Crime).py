
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
import seaborn as sns

crime=pd.read_csv('C:/Users/Biswa/Desktop/360DigiTMG/13 . Module 13/Assignmest/Data Set/crime_data.csv')

def norm_fucn(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

crime_norm=norm_fucn(crime.iloc[:,1:5])

k=list(range(2,8))
k
TWSS=[]
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(crime_norm)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(crime_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,crime_norm.shape[1]),'euclidean')))
    TWSS.append(sum(WSS))

plt.plot(k,TWSS,'ro-')
plt.xlabel("No of clusters")
plt.ylabel("Total within ss")
plt.xticks(k)
plt.title('Elbow Curve')  
plt.show() 

model=KMeans(n_clusters=3)
model.fit(crime_norm)
model.labels_
md=pd.Series(model.labels_)
crime['Clust']=md
crime.rename(columns={'Unnamed: 0':'State'},inplace=True)

cm=crime.groupby('Clust').mean()
cm=cm.reset_index()

f,ax=plt.subplots(figsize=(12,8))
sns.barplot(y='State',x='Murder',data=crime,errwidth=5,hue='Clust',dodge=False)
plt.show()

f,ax=plt.subplots(figsize=(12,8))
sns.barplot(y='State',x='Assault',data=crime,errwidth=5,hue='Clust',dodge=False)
plt.show()

f,ax=plt.subplots(figsize=(12,8))
sns.barplot(y='State',x='Rape',data=crime,errwidth=5,hue='Clust',dodge=False)
plt.show()

f,ax=plt.subplots(figsize=(12,6))
sns.barplot(x='Clust',y='Murder',data=cm,errwidth=5,hue='Clust',dodge=False)
plt.show()

f,ax=plt.subplots(figsize=(12,6))
sns.barplot(x='Clust',y='Assault',data=cm,errwidth=5,hue='Clust',dodge=False)
plt.show()

f,ax=plt.subplots(figsize=(12,6))
sns.barplot(x='Clust',y='UrbanPop',data=cm,errwidth=5,hue='Clust',dodge=False)
plt.show()

cluster0=crime[crime['Clust']==0]
cluster1=crime[crime['Clust']==1]
cluster2=crime[crime['Clust']==2]
