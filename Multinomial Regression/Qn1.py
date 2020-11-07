import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

prog=pd.read_csv('mdata.csv')
prog=prog.drop(columns=['Unnamed: 0','id'],axis=1)

prog.shape
prog.describe()
prog.isnull().sum()
prog.info()
prog.columns

# Segregating Categorical & Numeric Categoties

categorical=prog.select_dtypes(include=object)
numeric=prog.select_dtypes(exclude=object)

# Exploratory data analysis

 # Categorical Columns

for i in categorical.columns:
    print(categorical[i].value_counts())
    print()
   
fig,((ax1,ax2,ax3),(ax4,ax5,ax6))=plt.subplots(2,3,figsize=(14,8))
sns.countplot(x='female',data=categorical,ax=ax1)
sns.countplot(x='ses',data=categorical,ax=ax2)
sns.countplot(x='schtyp',data=categorical,ax=ax3)
sns.countplot(x='prog',data=categorical,ax=ax4)
sns.countplot(x='honors',data=categorical,ax=ax5)
plt.show()

  # Males & females distribution in different aspects

sns.catplot(x='schtyp',hue='female',data=categorical,kind='count',legend=False)
plt.xlabel('School Type')
plt.legend(loc='best')
plt.show()

sns.catplot(x='ses',hue='female',kind='count',data=categorical,legend=False)
plt.title('Socioeconomic Class')
plt.xlabel('Economic Type')
plt.legend(loc='best')
plt.show()

sns.catplot(x='prog',hue='female',kind='count',data=categorical,legend=False)
plt.xlabel('Type of program')
plt.legend(loc='best')
plt.show()

sns.catplot(x='honors',hue='female',kind='count',data=categorical,legend=False)
plt.xlabel('Enrollment status')
plt.legend(loc='best')
plt.show()

 # Numerical Columns

import scipy.stats as stats
import pylab

for i in numeric.columns:
    stats.probplot(numeric[i],dist='norm',plot=pylab)
    plt.show()

for i in numeric.columns:
    skew=numeric[i].skew()
    sns.distplot(numeric[i],label='Skewness= %0.2f' %(skew))
    plt.legend(loc='best')
    plt.show()

 # Bi-variate analysis
# Program Vs scores in different tests
sns.catplot(x='prog',y='read',hue='female',kind='bar',data=prog)
sns.catplot(x='prog',y='read',hue='female',kind='bar',data=prog)
sns.catplot(x='prog',y='write',hue='female',kind='bar',data=prog)
sns.catplot(x='prog',y='math',hue='female',kind='bar',data=prog)
sns.catplot(x='prog',y='science',hue='female',kind='bar',data=prog)

sns.catplot(x='prog',y='read',hue='female',kind='box',data=prog)
sns.stripplot(x='prog',y='read',hue='female',color='k',data=prog,jitter=0.2,alpha=0.4)
plt.show()
sns.catplot(x='prog',y='write',hue='female',kind='box',data=prog)
sns.stripplot(x='prog',y='write',hue='female',data=prog,jitter=0.2,alpha=0.4,color='k')
plt.show()
sns.catplot(x='prog',y='math',hue='female',kind='box',data=prog)
sns.stripplot(x='prog',y='math',hue='female',data=prog,jitter=0.2,alpha=0.4,color='k')
plt.show()
sns.catplot(x='prog',y='science',hue='female',kind='box',data=prog)
sns.stripplot(x='prog',y='science',hue='female',data=prog,jitter=0.2,alpha=0.4,color='k')
plt.show()

# Label Encodong  
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()  
prog['female']=lb.fit_transform(prog['female'])
prog['ses']=lb.fit_transform(prog['ses'])
prog['schtyp']=lb.fit_transform(prog['schtyp'])
prog['prog']=lb.fit_transform(prog['prog'])
prog['honors']=lb.fit_transform(prog['honors'])

  # Corelation Heatmap
corr=prog.corr()
sns.heatmap(corr,cmap='coolwarm',annot=True) 

# Pair plot
sns.pairplot(prog)
plt.show()

# Segregating feature & target column
x=prog.drop('prog',axis=1)
y=prog.iloc[:,3]


# Feature selection
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(x,y)
importance=pd.Series(model.feature_importances_,index=x.columns)
importance=pd.DataFrame(importance).reset_index()
importance.columns=['Features','Scores']

sns.catplot(x='Scores',y='Features',data=importance,kind='bar',aspect=1.5)
plt.show()

# Model Building

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression 
regressor=LogisticRegression(multi_class='multinomial',solver='newton-cg')
regressor.fit(x_train,y_train)
from sklearn.metrics import accuracy_score
y_pred=regressor.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)

from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(y_test,y_pred)
cm

report=classification_report(y_test,y_pred)
print(report)

#Model Building with feature selection
xf=x.iloc[:,3:7]
yf=y
xf_train,xf_test,yf_train,yf_test=train_test_split(xf,yf,test_size=0.3,random_state=0)
regressor.fit(xf_train,yf_train)
yf_pred=regressor.predict(xf_test)
accuracy_f=accuracy_score(yf_test,yf_pred)
cm=confusion_matrix(yf_test,yf_pred)
cm
report=classification_report(yf_test,yf_pred)
print(report)

