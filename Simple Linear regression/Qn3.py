import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Importing the data :
emp=pd.read_csv('emp_data.csv')
emp.head()

#Checking column names:
c=1
for i in emp.columns:
    print(c,'.',i)
    c+=1

#Checking null values : 
emp.info()
emp.isnull().sum()

#Segregating mput and output data:
x=emp.iloc[:,0].values
y=emp.iloc[:,1].values

#Checking Correlation :
emp.corr()
sns.pairplot(emp)

#EDA : 
emp.describe()

for i in emp.columns:
    print('Mean for {} is {}'.format(i,emp[i].mean()))
    print('Median for {} is {}'.format(i,emp[i].median()))
    print('Mode for {} is {}'.format(i,emp[i].mode()))
    
   #Univariate Analysis for numerical data :
for i in emp.columns:
    print('column:',i)
    print('Skewness for {} is {}'.format(i,emp[i].skew()))
    print('Kurtosis for{} is {}'.format(i,emp[i].kurt()))
    sns.distplot(emp[i])
    plt.show
     
   #Violin Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.violinplot(emp['Salary_hike'],emp['Churn_out_rate'])
plt.show()

   #Box Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.boxplot(emp['Salary_hike'],emp['Churn_out_rate'])
plt.show()

   #Boxen Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.boxenplot(emp['Salary_hike'],emp['Churn_out_rate'])
plt.show()

#Model building 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=150)

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

np.log(emp['Salary_hike']).skew()
np.log(emp['Churn_out_rate']).skew()
plt.hist(np.log(emp['Salary_hike']))
plt.hist(np.log(emp['Churn_out_rate']))

emp_log=np.log(emp)

xl=emp_log.iloc[:,0].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(xl,y,test_size=0.4,random_state=150)

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

yexp=emp_log.iloc[:,1].values

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





