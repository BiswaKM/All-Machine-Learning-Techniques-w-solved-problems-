import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# 1 - Preparing problem
# A - Data Loading
glass=pd.read_csv('C:\\Users\\Biswa\Desktop\\360DigiTMG\\18 . Module 18 - Machine Learning K-Nearest Neighbour\\Assignment\\Dataset\\glass.csv')

# B - Loading and exploring the shape of dataset
glass.shape
glass.head()
glass.info()
glass.describe()
glass.isnull().sum()
glass.nunique()
glass['Type'].value_counts()

# 2 - Exploratory Data Analysis

 # A - Checking the normality of Data
import scipy.stats as stats
import pylab 

for i in glass.columns:
    f,ax=plt.subplots(figsize=(12,6))
    stats.probplot(glass[i],dist='norm',plot=pylab)
    plt.show()

 # B - Univariate plots
for i in glass.columns:
    skew=glass[i].skew()
    sns.distplot(glass[i],kde=False,label='Skew=%.2f' %(skew),bins=30)
    plt.legend(loc='best')
    plt.show()

 #Box Plots:
 
 #Grouped plot
fig=plt.figure(figsize=(10,5))
glass.boxplot(column=['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type'])
plt.show()

 #Individual Box plots
for i in glass.columns:
    f,ax=plt.subplots(figsize=(12,6))
    ax=sns.boxplot(glass[i])
    plt.show()
    
 # MultiVariate plots

 # A - Corelation heatmap
corr=glass.corr()
f,ax=plt.subplots(figsize=(8,8))
heat_map=sns.heatmap(corr,annot=True,cmap='RdYlGn')
plt.xticks(range(len(corr.columns)))
plt.yticks(range(len(corr.columns)))
plt.show()

 # Distribution of target column
sns.countplot(glass['Type'])
plt.show()

for i in glass.columns:
    f,ax=plt.subplots(figsize=(6,4))
    sns.violinplot(x='Type',y=glass[i],data=glass)
    plt.show()

 # B - Pair plot
sns.pairplot(glass)
plt.show()
   
# Segregating input and output features
x=glass.iloc[:,:-1]
y=glass.iloc[:,-1]

# 3 - Feature Engineering

# Feature Importance
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(x,y)
feature_importance=pd.Series(model.feature_importances_,index=x.columns)
feature_importance=pd.DataFrame(feature_importance)
feature_importance.columns=['Importance']
feature_importance=feature_importance.reset_index()

feature_score=sns.catplot(y='index',x='Importance',data=feature_importance,kind='bar',aspect=2)
plt.show()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x)
x=sc.transform(x)

# 4 - Model Building 

# Finding Optimal K value
from sklearn.neighbors import KNeighborsClassifier   
from sklearn.model_selection import cross_val_score
accuracy_score=[]
for i in range(1,40,2):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x,y,cv=10,)
    accuracy_score.append(score.mean())
    
 # Plotting K VS Accuracy 
f,ax=plt.subplots(figsize=(12,6))
plt.plot(range(1,40,2),accuracy_score,marker='o',color='r')
plt.xlabel('K Number')
plt.ylabel('Accuracy')
plt.title('Accuracy rate')
plt.show()

# Modeling with k=3

knn_f=KNeighborsClassifier(n_neighbors=3)
score=cross_val_score(knn_f,x,y,cv=10)

score_mean=score.mean()
score_max=score.max()
score_min=score.min()

# Confusion Matrix & Classification Report
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=0)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

ytrain_pred=knn.predict(x_train)
ytest_pred=knn.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,ytest_pred)
cm

from sklearn.metrics import classification_report 
report=classification_report(y_test,ytest_pred)
print(report)

