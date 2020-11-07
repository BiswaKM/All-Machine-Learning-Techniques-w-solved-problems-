import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as stats
coro=pd.read_csv('ToyotaCorolla.csv',encoding='unicode_escape')
coro.info()
coro_f1=coro.drop(columns=['Id','Model','Mfg_Month','Mfg_Year','Fuel_Type','Met_Color','Color','Automatic','Cylinders','Mfr_Guarantee','BOVAG_Guarantee','Guarantee_Period','ABS','Airbag_1','Airbag_2','Airco','Automatic_airco','Boardcomputer','CD_Player','Central_Lock','Powered_Windows','Power_Steering','Radio','Mistlamps','Sport_Model','Backseat_Divider','Metallic_Rim','Radio_cassette','Tow_Bar'])
#Checking Corelation
sns.pairplot(coro_f1)
corrvalue=coro_f1.corr()
coro_f1.describe()

#Segregating input and output data
x=coro_f1.iloc[:,1:]
y=coro_f1.iloc[:,0:1]

#Segregating numerical & categorical data
c=1
print('Numerical columns are:')
for i in coro_f1.columns:
    numerical=coro_f1.select_dtypes(exclude=object)
    print(c,'.',i)
    c+=1

#EDA:

 #Distribution of data
for i in coro_f1.columns:
    f,ax=plt.subplots(figsize=(12,5))
    sns.distplot(coro[i],kde=False,fit=stats.gamma)
    plt.show()

 #Box plot
for i in coro_f1.columns:
    f,ax=plt.subplots(figsize=(12,5))
    sns.boxplot(coro_f1[i])
    plt.show()

 #Variation of price with gears
f,ax=plt.subplots(figsize=(12,5))
sns.boxplot(x='Gears',y='Price',data=coro_f1)
plt.show()
    
lp=np.log(coro_f1['Price'])
coro_f1=coro_f1.astype(int)
for i in coro_f1.columns:
    sns.jointplot(x=coro_f1[i],y=lp,data=coro_f1)
    plt.show()

sns.jointplot(x=coro_f1['Gears'],y=coro_f1['Price'])

 #Price Variation with HP
grouped_hp=coro_f1.groupby('HP')['Price'].aggregate(np.median).reset_index()
plt.figure(figsize=(12,5))
sns.pointplot(x=grouped_hp['HP'].values,y=grouped_hp['Price'].values)
plt.ylabel('Median hp')
plt.xlabel('Median Price')
plt.show()

 #Price Variation with cc
grouped_cc=coro_f1.groupby('cc')['Price'].aggregate(np.median).reset_index()
sns.pointplot(grouped_cc['cc'],grouped_cc['Price'])
plt.show()
 #Price Variation with age 
plt.figure(figsize=(12,20))
grouped_age=coro_f1.groupby('Age_08_04')['Price'].aggregate(np.median).reset_index()
chart=sns.pointplot(grouped_age['Age_08_04'],grouped_age['Price'])
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='right')
plt.show()

 #Price variation with gears
grouped_gears=coro_f1.groupby('Gears')['Price'].aggregate(np.median).reset_index()
sns.pointplot(grouped_gears['Gears'],grouped_gears['Price'],rotation=45)
plt.show()

#Model building
x=coro_f1.iloc[:,1:]
y=coro_f1.iloc[:,0:1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

from sklearn.metrics import r2_score 
score=r2_score(y_test,y_pred)

from math import sqrt
from sklearn.metrics import mean_squared_error
rmse=sqrt(mean_squared_error(y_test,y_pred))

#Checking VIF score
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=[variance_inflation_factor(x.values,i)for i in range(x.shape[1])]
vif=pd.DataFrame({'VIF':vif},index=(x.columns))

x_vif=x.drop(columns=['Weight','Gears','HP','cc','Age_08_04'])
vif=[variance_inflation_factor(x_vif.values,i)for i in range(x_vif.shape[1])]
vif_n=pd.DataFrame({'VIF':vif},index=(x_vif.columns))

#Feature Selection with univariate selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures=SelectKBest(score_func=chi2,k=5)
fit=bestfeatures.fit(x,y)
df_score=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x.columns)
featurescores=pd.concat([dfcolumns,df_score],axis=1)
featurescores.columns=['Specs','Score']
print('Feature Scores are : \n',featurescores)

#Plotting important features
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(x,y)

feach_imp=pd.Series(model.feature_importances_,index=x.columns)
feach_imp.plot(kind='barh')
plt.show()

#Modeling with important features

x_f=x.drop(columns=['HP','cc','Doors','Gears'])
y_f=np.log(y)

x_train,x_test,y_train,y_test=train_test_split(x_f,y_f,test_size=0.3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

score_f=r2_score(y_test,y_pred)
rmse_f=sqrt(mean_squared_error(y_test,y_pred))

x_f.describe()
np.exp(y_f).describe()

#Modeling with Log transformation
x_l=np.log(x_f)
y_l=y_f

x_train,x_test,y_train,y_test=train_test_split(x_l,y_l,test_size=0.3,random_state=0)
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

score_l=r2_score(y_test,y_pred)
rmse_l=sqrt(mean_squared_error(y_test,y_pred))

x_f.describe()
np.exp(y_f).describe()











