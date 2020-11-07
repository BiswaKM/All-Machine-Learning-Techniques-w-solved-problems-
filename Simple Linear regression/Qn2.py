import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Importing the data :
dt=pd.read_csv('delivery_time.csv')
dt.head()

#Checking column names:
c=1
for i in dt.columns:
    print(c,'.',i)
    c+=1

#Checking null values : 
dt.info()
dt.isnull().sum()

#Segregating mput and output data:
x=dt.iloc[:,1].values
y=dt.iloc[:,0].values

#Checking Correlation :
dt.corr()
plt.scatter(np.log(dt['Sorting Time']),dt['Delivery Time'])
plt.scatter(dt['Sorting Time'],np.log(dt['Delivery Time']))
sns.pairplot(dt)

#EDA : 
dt.describe()

for i in dt.columns:
    print('Mean for {} is {}'.format(i,dt[i].mean()))
    print('Median for {} is {}'.format(i,dt[i].median()))
    print('Mode for {} is {}'.format(i,dt[i].mode()))
    
   #Univariate Analysis for numerical data :
for i in dt.columns:
    print('column:',i)
    print('Skewness for {} is {}'.format(i,dt[i].skew()))
    print('Kurtosis for{} is {}'.format(i,dt[i].kurt()))
    sns.distplot(dt[i])
    plt.show
     
   #Violin Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.violinplot(dt['Sorting Time'],dt['Delivery Time'])
plt.show()

   #Box Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.boxplot(dt['Sorting Time'],dt['Delivery Time'])
plt.show()

   #Boxen Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.boxenplot(dt['Sorting Time'],dt['Delivery Time'])
plt.show()

#Model building 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=25)

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

np.log(dt['Sorting Time']).skew()
np.log(dt['Delivery Time']).skew()
plt.hist(np.log(dt['Sorting Time']))
plt.hist(np.log(dt['Delivery Time']))

dt_log=np.log(dt)

xl=dt_log.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(xl,y,test_size=0.4,random_state=25)

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

yexp=dt_log.iloc[:,0].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,yexp,test_size=0.4,random_state=25)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))

y_pred=regressor.predict(x_test.reshape(-1,1))

from sklearn.metrics import r2_score
score_ex=r2_score(y_test,y_pred)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_exp=sqrt(mean_squared_error(y_test,yl_pred))





