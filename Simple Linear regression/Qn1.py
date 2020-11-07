import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

#Importing the data :
wg=pd.read_csv('calories_consumed.csv')
wg.head()

#Checking column names:
c=1
for i in wg.columns:
    print(c,'.',i)
    c+=1

#Checking null values : 
wg.info()
wg.isnull().sum()

#Segregating mput and output data:
x=wg.iloc[:,1].values
y=wg.iloc[:,0].values

#Checking Correlation :
wg.corr()
plt.scatter(np.log(wg['Calories Consumed']),wg['Weight gained (grams)'])
plt.scatter(wg['Calories Consumed'],np.log(wg['Weight gained (grams)']))
sns.pairplot(wg)

#EDA : 
wg.describe()

for i in wg.columns:
    print('Mean for {} is {}'.format(i,wg[i].mean()))
    print('Median for {} is {}'.format(i,wg[i].median()))
    print('Mode for {} is {}'.format(i,wg[i].mode()))
    
   #Univariate Analysis for numerical data :
for i in wg.columns:
    print('column:',i)
    print('Skewness for {} is {}'.format(i,wg[i].skew()))
    print('Kurtosis for{} is {}'.format(i,wg[i].kurt()))
    sns.distplot(wg[i])
    plt.show
     
   #Violin Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.violinplot(wg['Calories Consumed'],wg['Weight gained (grams)'])
plt.show()

   #Box Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.boxplot(wg['Calories Consumed'],wg['Weight gained (grams)'])
plt.show()

   #Boxen Plot :
f,ax=plt.subplots(figsize=(7,5))
sns.boxenplot(wg['Calories Consumed'],wg['Weight gained (grams)'])
plt.show()

#Model building 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

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

np.log(wg['Calories Consumed']).skew()
np.log(wg['Weight gained (grams)']).skew()
plt.hist(np.log(wg['Calories Consumed']))
plt.hist(np.log(wg['Weight gained (grams)']))

wg_log=np.log(wg)

xl=wg_log.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(xl,y,test_size=0.3,random_state=0)

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

yexp=wg_log.iloc[:,0].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,yexp,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train.reshape(-1,1),y_train.reshape(-1,1))

y_pred=regressor.predict(x_test.reshape(-1,1))

from sklearn.metrics import r2_score
score_ex=r2_score(y_test,y_pred)
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_exp=sqrt(mean_squared_error(y_test,yl_pred))





