import math
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#importing Dataset
comp=pd.read_csv('Computer_Data.csv')
comp.info()

#Droping Index column
comp=comp.drop('Unnamed: 0',axis=1)

#Checking Null values
comp.info()
comp.isnull().sum()

#EDA
comp.corr()
sns.pairplot(comp)

  #Segregating Categorial data and Numeric data
numerical=comp.select_dtypes(exclude=object)
print('Numerical columns are :')
c=1
for i in numerical.columns:
    print(c,'.',i)
    c+=1

categorical=comp.select_dtypes(include=object)
print('Categorical columns are :')
c=1
for i in categorical.columns:
    print(c,'.',i)
    c+=1

numerical.describe()

  #Univariate analysis for categorial data
for i in categorical.columns:
    print('Column:',i)
    print(comp[i].value_counts())
    f,ax=plt.subplots(figsize=(12,5))
    sns.countplot(comp[i])
    plt.show()

  #Univariate analysis for numerical data
for i in numerical.columns:
    print('column:',i)
    print('skewness of {} is {}'.format(i,comp[i].skew()))
    f,ax=plt.subplots(figsize=(12,5))
    sns.distplot(comp[i],kde=False,fit=stats.gamma)
    plt.show()

  #Bivariate analysis
   #Box plots
for i in categorical.columns:
    for j in numerical.columns:
        f,ax=plt.subplots(figsize=(12,5))
        sns.boxplot(comp[i],comp[j])
        plt.show()
    #joint plots    
for i in numerical.columns:
    sns.jointplot(x='price',y=comp[i],kind='kde',color='k',data=comp)
    plt.show()
    #Scatter plots
for i in numerical.columns:
    for j in categorical.columns:
        sns.scatterplot(x='price',y=comp[i],hue=comp[j],data=comp)
        plt.show()
        
#Converting categorical data into numeric
comp['cd']=comp['cd'].eq('yes').astype(int)
comp['multi']=comp['cd'].eq('yes').astype(int)
comp['premium']=comp['premium'].eq('yes').astype(int)

#Segregatingi mput and output data
x=comp.iloc[:,1:]
y=comp.iloc[:,:1]

#Spliting the dataset into test and train set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#fitting multiple linear regression into training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)
#Predicting the test set reasult
y_pred=regressor.predict(x_test)

#R2 value
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)

#RMSE value
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(y_test,y_pred))

#Checking VIF values
from statsmodels.stats.outliers_influence import variance_inflation_factor
numerical_vif=numerical.iloc[:,1:]
vif=[variance_inflation_factor(numerical_vif.values,i)for i in range(numerical_vif.shape[1])]
vif=pd.DataFrame({'VIF':vif},index=numerical_vif.columns)
print(vif)

#Feature Selection using Univariate Selection and Corelation Matrix with Heatmap:

x=x.astype(int)
y=y.astype(int)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures=SelectKBest(score_func=chi2,k=4)
fit=bestfeatures.fit(x,y)
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)
featureScores=pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns=['Department','Scores']
print('Feature Scores are : \n',featureScores)

#Modeling after selection:
x_n=comp.drop(['cd','multi','screen','premium'],axis=1)
y_n=comp.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_n,y_n,test_size=0.3,random_state=0)

#fitting multiple linear regression into training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set reasult
y_pred=regressor.predict(x_test)

#Checking multiple R2 value
from sklearn.metrics import r2_score
Score_n=r2_score(y_test,y_pred)

#RMSE value
from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE_f=np.sqrt(mean_squared_error(y_test,y_pred))

input_paramiters=x_n.describe()























