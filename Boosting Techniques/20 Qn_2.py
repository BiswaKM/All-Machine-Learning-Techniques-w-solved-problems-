import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df=pd.read_csv('wbcd.csv')
df=df.drop('id',axis=1)

df.shape
df.info()
df.describe()
df.isnull().sum()

x=df.iloc[:,1:]
y=df.iloc[:,0]

# Univariate selection
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.feature_selection import f_classif 
best_features=SelectKBest(score_func=f_classif,k=30)    
fit=best_features.fit(x,y)    
dfscore=pd.DataFrame(fit.scores_,index=x.columns)   

dfscore.plot(kind='barh')   

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y.iloc[:]=le.fit_transform(y.iloc[:])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x.iloc[:,:]=sc.fit_transform(x.iloc[:,:])

# Feature Importance
from sklearn.tree import ExtraTreeClassifier
classifier=ExtraTreeClassifier()
classifier.fit(x,y)
importance=pd.Series(classifier.feature_importances_,index=x.columns)
importance.plot(kind='barh')

# Segregating training & testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# Modelling
# Extreme Gradient Boosting:
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
gb=BaggingClassifier(DecisionTreeClassifier(),n_estimators=20,max_samples=0.5,max_features=1)
gb.fit(x_train,y_train)
gb.score(x_train,y_train)
gb.score(x_test,y_test)

# Adaboost
from sklearn.ensemble import AdaBoostClassifier
ada=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=20,learning_rate=1)
ada.fit(x_train,y_train)
ada.score(x_train,y_train)
ada.score(x_test,y_test)

# Voting classifier
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



























