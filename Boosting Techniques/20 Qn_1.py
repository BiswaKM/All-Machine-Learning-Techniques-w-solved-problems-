import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df=pd.read_csv('Diabetes_RF.csv')

df.shape
df.info()
df.describe()
df.isnull().sum()

x=df.iloc[:,:-1]
y=df.iloc[:,-1]

# Univariate feature selection
from sklearn.feature_selection import SelectKBest,f_classif
best_features=SelectKBest(score_func=f_classif,k=8)
best_features.fit(x,y)
df_score=pd.DataFrame(best_features.fit(x,y).scores_,index=x.columns)
df_score.plot(kind='barh')

# EDA
plt.figure()
sns.pairplot(df,hue=' Class variable',palette='plasma')
plt.show()

plt.figure()
corr=df.corr()
sns.heatmap(corr,annot=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x.iloc[:,:]=sc.fit_transform(x.iloc[:,:])

# Label Encoder
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y.iloc[:]=le.fit_transform(y.iloc[:])

# feature Importance
from sklearn.tree import ExtraTreeClassifier
classifier=ExtraTreeClassifier()
classifier.fit(x,y)
importance=pd.Series(classifier.feature_importances_,index=x.columns)
importance.plot(kind='barh')

#Spliting Train & Test Data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# Modelling
# Extreme Gradint Boosting:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bg=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1,n_estimators=20)
bg.fit(x_train,y_train)
bg.score(x_train,y_train)
bg.score(x_test,y_test)

# Adaboost:    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
ada=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=20,learning_rate=1)    
ada.fit(x_train,y_train)    
ada.score(x_train,y_train)
ada.score(x_test,y_test)

# Voting Classifier:
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
lr=LogisticRegression()
dt=DecisionTreeClassifier()
svm=SVC(kernel='poly',degree=2)

eve=VotingClassifier(estimators=[('lr',lr),('dt',dt),('svm',svm)],voting='hard')    
eve.fit(x_train,y_train)    
eve.score(x_train,y_train)    
eve.score(x_test,y_test)    

# Model with univariate feature selection
x.columns
xf=x[[' Plasma glucose concentration',' Body mass index',' Age (years)']]    
yf=y  
xf_train,xf_test,yf_train,yf_test=train_test_split(xf,yf,test_size=0.2,random_state=0)    
# Gradient Boost
bg.fit(xf_train,yf_train)
bg.score(xf_train,yf_train)
bg.score(xf_test,yf_test)
# Ada Boost
ada.fit(xf_train,yf_train)    
ada.score(xf_train,yf_train)    
ada.score(xf_test,yf_test)    
# Voting classifier
eve.fit(xf_train,yf_train)
eve.score(xf_train,yf_train)
eve.score(xf_test,yf_test)
