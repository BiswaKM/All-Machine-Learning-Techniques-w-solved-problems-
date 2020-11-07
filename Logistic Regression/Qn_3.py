import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df=pd.read_csv('ToyotaCorolla.csv',encoding='unicode_escape')
df=df.drop(columns=['Model','Id'])

df.shape
df.info()
df.describe()
df.isnull().sum()

# EDA

#Segregating numerical & categorical data
c=1
print('Numerical columns are:')
for i in df.columns:
    numerical=df.select_dtypes(exclude=object)
    print(c,'.',i)
    c+=1

#EDA:
import scipy.stats as stats
 #Box plot
for i in df.select_dtypes(exclude=object).columns:
    f,ax=plt.subplots(figsize=(12,5))
    sns.boxplot(df[i])
    plt.show()

 #Variation of price with gears
f,ax=plt.subplots(figsize=(12,5))
sns.boxplot(x='Gears',y='Price',data=df)
plt.show()

lp=np.log(df['Price'])

for i in df.select_dtypes(exclude=object).columns:
    sns.jointplot(x=df[i],y=lp,data=df)
    plt.show()

sns.jointplot(x=df['Gears'],y=df['Price'])

 #Price Variation with HP
grouped_hp=df.groupby('HP')['Price'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,5))
sns.pointplot(x=grouped_hp['HP'].values,y=grouped_hp['Price'].values)
plt.ylabel('Median hp')
plt.xlabel('Median Price')
plt.show()

 #Price Variation with cc
grouped_cc=df.groupby('cc')['Price'].aggregate(np.median).reset_index()
sns.pointplot(grouped_cc['cc'],grouped_cc['Price'])
plt.show()
 #Price Variation with age 
plt.figure(figsize=(12,20))
grouped_age=df.groupby('Age_08_04')['Price'].aggregate(np.median).reset_index()
chart=sns.pointplot(grouped_age['Age_08_04'],grouped_age['Price'])
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.show()

 #Price variation with gears
grouped_gears=df.groupby('Gears')['Price'].aggregate(np.median).reset_index()
sns.pointplot(grouped_gears['Gears'],grouped_gears['Price'],rotation=45)
plt.show()

# Feature Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
for i in df.select_dtypes(include=object).columns:
    df[i]=le.fit_transform(df[i])

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df.iloc[:,:]=sc.fit_transform(df.iloc[:,:])

# Feature Selection
 # univariate Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression,chi2
best_features=SelectKBest(score_func=f_regression,k='all')
best_features.fit(df.iloc[:,1:],df.iloc[:,0])
feature_scores=pd.DataFrame(best_features.scores_,index=df.iloc[:,1:].columns)
feature_scores.plot(kind='barh')

 # Feature Selection
from sklearn.tree import ExtraTreeRegressor
regressor=ExtraTreeRegressor()
regressor.fit(df.iloc[:,1:],df.iloc[:,0])
importance_score=pd.Series(regressor.feature_importances_,index=df.iloc[:,1:].columns)
importance_score.plot(kind='barh')

# Segregating feature & target columns
x=df.iloc[:,1:]
y=df.iloc[:,0]

# Modelling
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

# Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from math import sqrt
from sklearn.metrics import mean_squared_error
ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=10)
ridge_regressor.fit(x,y)
print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

ridge_regressor.fit(x_train,y_train)
pred_ridge_train=ridge_regressor.predict(x_train)
ridge_score_train=r2_score(y_train,pred_ridge_train)
pred_ridge_test=ridge_regressor.predict(x_test)
ridge_score_test=r2_score(y_test,pred_ridge_test)
ridge_rmse=sqrt(mean_squared_error(y_test,pred_ridge_test))

# Lasso Regression
from sklearn.linear_model import Lasso
lasso=Lasso()
lasso_regression=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error')
lasso_regression.fit(x,y)
print(lasso_regression.best_params_)
print(lasso_regression.best_score_)
lasso_regression.fit(x_train,y_train)
pred_lasso_train=lasso_regression.predict(x_train)
lasso_score_train=r2_score(y_train,pred_lasso_train)
pred_lasso_test=lasso_regression.predict(x_test)
lasso_score_test=r2_score(y_test,pred_lasso_test)
lasso_rmse=sqrt(mean_squared_error(y_test,pred_lasso_test))






