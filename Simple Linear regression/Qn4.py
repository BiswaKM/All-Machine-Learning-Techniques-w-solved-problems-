import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Importing the data :
exp=pd.read_csv('Salary_Data.csv')
exp.head()

#Checking column names:
c=1
for i in exp.columns:
    print(c,'.',i)
    c+=1

#Checking null values : 
exp.info()
exp.isnull().sum()

#Segregating mput and output data:
x=exp.iloc[:,0].values
y=exp.iloc[:,1].values

#Checking Correlation :
exp.corr()
sns.pairplot(exp)

#EDA : 
exp.describe()

for i in exp.columns:
    print('Mean for {} is {}'.format(i,exp[i].mean()))
    print('Median for {} is {}'.format(i,exp[i].median()))
    print('Mode for {} is {}'.format(i,exp[i].mode()))
    
   #Univariate Analysis for numerical data :
for i in exp.columns:
    print('column:',i)
    print('Skewness for {} is {}'.format(i,exp[i].skew()))
    print('Kurtosis for{} is {}'.format(i,exp[i].kurt()))
     
   #Violin Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.violinplot(exp['YearsExperience'],exp['Salary'])
plt.show()

   #Box Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.boxplot(exp['YearsExperience'],exp['Salary'])
plt.show()

   #Boxen Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.boxenplot(exp['YearsExperience'],exp['Salary'])
plt.show()

#Model building 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))

y_pred=regressor.predict(x_test.reshape(-1,1))

from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=sqrt(mean_squared_error(y_test,y_pred))

#log transformation

np.log(exp['YearsExperience']).skew()
np.log(exp['Salary']).skew()
plt.hist(np.log(exp['YearsExperience']))
plt.hist(np.log(exp['Salary']))

exp_log=np.log(exp)

xl=exp_log.iloc[:,0].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(xl,y,test_size=0.4,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))

yl_pred=regressor.predict(x_test.reshape(-1,1))

from sklearn.metrics import r2_score
score_l=r2_score(y_test,yl_pred)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_log=sqrt(mean_squared_error(y_test,yl_pred))

#Exponential transformation

yexp=exp_log.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,yexp,test_size=0.4,random_state=150)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))

y_pred=regressor.predict(x_test.reshape(-1,1))

from sklearn.metrics import r2_score
score_ex=r2_score(y_test,y_pred)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_exp=sqrt(mean_squared_error(y_test,yl_pred))





