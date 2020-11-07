import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df=pd.read_csv('50_Startups.csv')

df.shape
df.info()
df.describe()
df.isnull().sum()

# EDA
numeric=df.select_dtypes(exclude=object)

import scipy.stats as stats
import pylab
for i in df.select_dtypes(exclude=object).columns:
    stats.probplot(df[i],dist='norm',plot=pylab)
    plt.show()
    
for i in df.select_dtypes(exclude=object).columns:
    skew=df[i].skew()
    sns.distplot(df[i],label='Skew=%0.2f' %skew)
    plt.legend(loc='best')
    plt.show()

df.select_dtypes(exclude=object).boxplot()
for i in df.select_dtypes(exclude=object).columns:
    sns.catplot(x='State',y=i,data=df,kind='bar')

cor=df.corr()
sns.heatmap(cor,annot=True)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()    
for i in df.select_dtypes(exclude=object):
    df[i]=sc.fit_transform(df[i].values.reshape(-1,1))
    
# Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()    
df['State']=le.fit_transform(df['State'])    
    
# Segregating features & target columns
x=df.drop('Profit',axis=1)
y=df[['Profit']]

# Feature Importance
 
 # Univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
best_features=SelectKBest(score_func=f_regression,k='all')
fit=best_features.fit(x,y)
importance_score=pd.DataFrame(fit.scores_,index=x.columns)
importance_score.plot(kind='barh')

 # Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)
feature_score=pd.Series(model.feature_importances_,index=x.columns)
feature_score.plot(kind='barh')

# Segregating train & test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)    

# Modelling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt

# Linear Regression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
score_linear=r2_score(y_test,y_pred)
rmse_linear=sqrt(mean_squared_error(y_test,y_pred))
sns.distplot(y_test-y_pred)

# Ridge Regresson
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(x,y)    
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)    
print(ridge_regressor.best_estimator_)    

ridge=Ridge(alpha=1e-15, copy_X=True, fit_intercept=True, max_iter=None,
      normalize=False, random_state=None, solver='auto', tol=0.001)
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(x_train,y_train)
pred_ridge=ridge_regressor.predict(x_test)
score_ridge=r2_score(y_test,pred_ridge)
rmse_ridge=sqrt(mean_squared_error(y_test,pred_ridge))
sns.distplot(y_test-pred_ridge)

# Lasso Regresson
from sklearn.linear_model import Lasso
lasso=Lasso()
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)
lasso_regressor.fit(x,y)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

lasso_regressor.fit(x_train,y_train)
pred_lasso=lasso_regressor.predict(x_test)
score_lasso=r2_score(y_test,pred_lasso)
rmse_lasso=sqrt(mean_squared_error(y_test,pred_lasso))

# Modelling after Feature Selection
x_new=x[['Marketing Spend','R&D Spend']]
y_new=y
xn_train,xn_test,yn_train,yn_test=train_test_split(x_new,y_new,test_size=0.2,random_state=0)    

ridge_regressor.fit(xn_train,yn_train)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)
pred_ridge_n=ridge_regressor.predict(xn_test)
score_ridge_n=r2_score(yn_test,pred_ridge_n)

lasso_regressor.fit(xn_train,yn_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)
pred_lasso_n=lasso_regressor.predict(xn_test)
score_lasso_n=r2_score(yn_test,pred_lasso_n)
















