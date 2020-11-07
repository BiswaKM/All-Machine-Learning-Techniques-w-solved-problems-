import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df=pd.read_csv('Computer_Data.csv')
df=df.drop('Unnamed: 0',axis=1)

df.shape
df.info()
df.describe()
df.isnull().sum()

# EDA
df.corr()
sns.pairplot(df)

  #Segregating Categorial data and Numeric data
numerical=df.select_dtypes(exclude=object)
print('Numerical columns are :')
c=1
for i in numerical.columns:
    print(c,'.',i)
    c+=1

categorical=df.select_dtypes(include=object)
print('Categorical columns are :')
c=1
for i in categorical.columns:
    print(c,'.',i)
    c+=1

numerical.describe()

  #Univariate analysis for categorial data
for i in categorical.columns:
    print('Column:',i)
    print(df[i].value_counts())
    f,ax=plt.subplots(figsize=(12,5))
    sns.countplot(df[i])
    plt.show()

  #Univariate analysis for numerical data
import scipy.stats as stats
for i in numerical.columns:
    print('column:',i)
    print('skewness of {} is {}'.format(i,df[i].skew()))
    f,ax=plt.subplots(figsize=(12,5))
    sns.distplot(df[i],kde=False,fit=stats.gamma)
    plt.show()

  #Bivariate analysis
   #Box plots
for i in categorical.columns:
    for j in numerical.columns:
        f,ax=plt.subplots(figsize=(12,5))
        sns.boxplot(df[i],df[j])
        plt.show()
    #joint plots    
for i in numerical.columns:
    sns.jointplot(x='price',y=df[i],kind='kde',color='k',data=df)
    plt.show()
    #Scatter plots
for i in numerical.columns:
    for j in categorical.columns:
        sns.scatterplot(x='price',y=df[i],hue=df[j],data=df)
        plt.show()

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in df.select_dtypes(include=object).columns:
    df[i]=le.fit_transform(df[i])
    
# Feature Importance
 # Univariate Selection
from sklearn.feature_selection import SelectKBest,f_regression
best_features=SelectKBest(score_func=f_regression,k='all')
fit=best_features.fit(df.iloc[:,1:],df.iloc[:,0])
feature_score=pd.DataFrame(fit.scores_,index=df.iloc[:,1:].columns)
feature_score.columns=['Importance Score']
feature_score.plot(kind='barh')

 # Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
regressor=ExtraTreesRegressor()
regressor.fit(df.iloc[:,1:],df.iloc[:,0])
importance_score=pd.Series(regressor.feature_importances_,index=df.iloc[:,1:].columns)
importance_score.plot(kind='barh')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df.iloc[:,0:5]=sc.fit_transform(df.iloc[:,0:5])
df[['ads','trend']]=sc.fit_transform(df[['ads','trend']])

# Segregating features & target columns
x=df.iloc[:,1:]
y=df.iloc[:,0]

# Modelling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso
import sklearn

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

# Linear Regression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.score(x_train,y_train)
regressor.score(x_test,y_test)
y_pred=regressor.predict(x_test)
score_linear_regression=r2_score(y_test,y_pred)
ramse_linear=sqrt(mean_squared_error(y_test,y_pred))

# Ridge Regression
ridge=Ridge()
sorted(sklearn.metrics.SCORERS.keys())
parameters={'alpha':[1e-15,1e-10,1e-8,1e-8,1e-3,1e-2,1,5,10,15,20,25,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=10)
ridge_regressor.fit(x,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

ridge_regressor.fit(x_train,y_train)
pred_ridge_train=ridge_regressor.predict(x_train)
ridge_score_train=r2_score(y_train,pred_ridge_train)
pred_ridge_test=ridge_regressor.predict(x_test)
ridge_score_test=r2_score(y_test,pred_ridge_test)
rmse_ridge=sqrt(mean_squared_error(y_test,pred_ridge_test))
sns.distplot(y_test-pred_ridge_test)

# Lasso Regression
lasso=Lasso()
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=10)
lasso_regressor.fit(x,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

lasso_regressor.fit(x_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
pred_lasso_train=lasso_regressor.predict(x_train)
lasso_score_train=r2_score(y_train,pred_lasso_train)
pred_lasso_test=lasso_regressor.predict(x_test)
lasso_score_test=r2_score(y_test,pred_lasso_test)
rmse_lasso=sqrt(mean_squared_error(y_test,pred_lasso_test))
sns.distplot(y_test-pred_lasso_test)

# Modeling with feature selection
xn=df[['ram','hd','speed','screen','trend','premium']]
yn=y
xn_train,xn_test,yn_train,yn_test=train_test_split(xn,yn,test_size=0.3,random_state=0)
ridge_regressor.fit(xn_train,yn_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
pred_ridge_n=ridge_regressor.predict(xn_test)
reidge_score_n=r2_score(yn_test,pred_ridge_n)












