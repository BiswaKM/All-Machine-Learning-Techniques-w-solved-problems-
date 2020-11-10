import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 

# Data Loading & Cheking shape of data
sales=pd.read_csv('Company_Data.csv')
sales.info()
sales.shape
sales.isnull().sum()
sales.describe()

# EDA
categorical=sales.select_dtypes(include='object')
numerical=sales.select_dtypes(exclude='object')

for i in numerical.columns:
    skew=numerical[i].skew()
    sns.distplot(numerical[i],label='Skewness=%0.2f'%skew)
    plt.legend(loc='best')
    plt.show()

import scipy.stats as stats
import pylab
for i in numerical.columns:
    stats.probplot(numerical[i],dist='norm',plot=pylab)
    plt.show()

#Converting target column into categorical
sales['Sales']=np.where(sales['Sales']>=9,'yes','No')

#Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sales['ShelveLoc']=le.fit_transform(sales['ShelveLoc'])
sales['Urban']=le.fit_transform(sales['Urban'])
sales['US']=le.fit_transform(sales['US'])
sales['Sales']=le.fit_transform(sales['Sales'])

# Segregating x & y
x=sales.iloc[:,1:]
y=sales.iloc[:,-0]

# Feature Scaling
from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
x.iloc[:,0:4]=sc.fit_transform(x.iloc[:,0:4])
x.iloc[:,6:8]=sc.fit_transform(x.iloc[:,6:8])

#Modeling
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

model=DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

score=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
print(report)


















