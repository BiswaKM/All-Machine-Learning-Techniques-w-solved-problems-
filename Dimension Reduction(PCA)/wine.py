import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

wine=pd.read_csv('wine.csv')
wine_f=wine.drop('Type',axis=1)
wine_f.head()

corrmat=wine.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(20,20))
g=sns.heatmap(wine[top_corr_features].corr(),annot=True,cmap='RdYlGn')

wine_f.info()
wine_f.isnull().sum()
wine_f.describe()

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(wine_f)
wine_f=sc.transform(wine_f)

#Applying PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=13)
pca_values=pca.fit_transform(wine_f)

#Variacne of each PCA '
var=pca.explained_variance_ratio_

var_cumi=np.cumsum(np.round(var,4)*100)
var_cumi
f,ax=plt.subplots(figsize=(12,6))
plt.plot(var_cumi)
plt.xlabel('PC Number')
plt.ylabel('Percentage Data')
plt.show()

#Selecting input variable
x=pca_values[:,:3]

#Hierarchial Clustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
f,ax=plt.subplots(figsize=(12,6))
dengrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

#Fitting hierarchial clustering to the dataset
from sklearn.cluster import AgglomerativeClustering

hc=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(wine)

cluster_labels=pd.Series(hc.labels_)
wine['Type_Hierarchical']=cluster_labels

#K-Means Clustering

 #Elbow plor for optimal number of cluster
from sklearn.cluster import KMeans
f,ax=plt.subplots(figsize=(12,6))
wcss=[]
k=list(range(1,11))
for i in k:
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(k,wcss,'ro-')
plt.title('The Elbow Plot')
plt.xlabel('No of Clusters')
plt.ylabel('WCSS')
plt.show()

#Fitting k-means to the dataset
kmeans=KMeans(n_clusters=3,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(wine)

cluster_labels=pd.Series(kmeans.labels_)
wine['Type_KMeans']=cluster_labels

KM_clusters=wine.groupby('Type_KMeans').mean()
HC_clusters=wine.groupby('Type_Hierarchical').mean()





















