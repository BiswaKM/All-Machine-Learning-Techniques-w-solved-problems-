import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy.stats as stats

#importing Dataset
startup=pd.read_csv('50_Startups.csv')
startup.describe()
startup.info()
startup.isnull().sum()

c=1
for i in startup.columns:
    print(c,'.',i)
    c+=1

#Correlation
startup.corr()
sns.pairplot(startup)

#EDA

 #Segregating numerical data and categorical data

 #Categorical Data
categorical=startup.iloc[:,-2]
numerical=startup.iloc[:,:-2]

categorical=startup.select_dtypes(include="object")
print("Categorical Columns :")
c=1
for i in categorical.columns:
    print(c,".",i)
    c+=1
 #Numerical Data
numerical=startup.select_dtypes(exclude="object")
print("Numerical Columns :")
c=1
for i in numerical:
    print(c,'.',i)
    c+=1

 #Checking categorical data distribution

for i in categorical.columns:
    print("columns:",i)
    print(startup[i].value_counts())
    f,ax=plt.subplots(figsize=(7,5))
    sns.countplot(startup[i])
    plt.show()
    print('\n')
    
 #Univariate analysis of numerical columns usinf distribution plot and checking the skewness

for i in numerical.columns:
    print('Column:',i)
    print('Skewness for {} is {}'.format(i,round(startup[i].skew(),3)))
    sns.distplot(startup[i])
    plt.show()

 #Bivariate analysis for categorical and numerical columns using Boxen plot,Box plot and Violin plot

 #Boxen Plot:
for i in categorical.columns:
    for j in numerical.columns:
        f,ax=plt.subplots(figsize=(12,5))
        sns.boxenplot(startup[i],startup[j])
        plt.show()

 #Box Plot:
for i in categorical.columns:
    for j in numerical.columns:
        f,ax=plt.subplots(figsize=(12,5))
        sns.boxplot(startup[i],startup[j])
        plt.show()

 #Violin Plot:
for i in categorical.columns:
    for j in numerical.columns:
        f,ax=plt.subplots(figsize=(12,5))
        sns.violinplot(startup[i],startup[j])
        plt.show()

f,ax=plt.subplots(figsize=(12,8))
sns.scatterplot(startup['R&D Spend'],startup['Administration'],hue=startup['State'])
plt.show()

f,ax=plt.subplots(figsize=(12,5))
sns.catplot(x='State',y='Profit',kind='bar',data=startup,label='Florida')
ax.legend()
plt.show()
        
 #Testing skewness and best tranformations

log=[]
sqrt=[]
for i in numerical.columns:
    print('checking skewness for',i)
    a=round(startup[i].skew(),3)
    print('1.Normal Distribution',a)
    d=pd.DataFrame()
    d['log']=np.log(startup[i])
    b=round(d['log'].skew(),3)
    print('Log transformation',b)
    d['sqrt']=np.sqrt(startup[i])
    c=round(d['sqrt'].skew(),3)
    print('sqrt transformation',c)
    print('the best transformation would be :')    
    if np.abs(c)<np.abs(a) and np.abs(c)<np.abs(b):
        print('sqrt transformation',c)
        sqrt.append(i)
    elif np.abs(b)<np.abs(a) and np.abs(b)<np.abs(c):
        print('log transformation',b)
        log.append(i)
    else :
        print('Normal',a)
    print('\n')
print('Features to under go log transformation are :',log)
print('Features to under go sqrt trasformation are :',sqrt)

#Checking VIF to drop individual predictors based on the inflation value

from statsmodels.stats.outliers_influence import variance_inflation_factor

numerical_vif=numerical.iloc[:,0:3]
vif=[variance_inflation_factor(numerical_vif.values,i) for i in range(numerical_vif.shape[1])]
vif=pd.DataFrame({'VIF':vif},index=numerical_vif.columns)
vif  

#Label Encoding :
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
startup['State']=le.fit_transform(startup['State'])

#Segregating mput and output data
x=startup.iloc[:,:4]
y=startup.iloc[:,4:5]

'''from sklearn.preprocessing import LabelEncoder
for i in startup.columns:
    le=LabelEncoder()
   startup[i]=le.fit_transform(startup[i])

Logit Model for the data
def norm_func(i):
    n=(i-i.min())/(i.max()-i.min())
    return n
y=norm_func(y)

import statsmodels.api as sm
xc=sm.add_constant(x)
model=sm.Logit(y,xc).fit()
model.summary()

#Creating Dummy Variable
states=pd.get_dummies(x['State'],drop_first=True)
Droping categorical column
x=x.drop(['State'],axis=1)
Concatinating
x=pd.concat([x,states],axis=1)'''
#Model Building

#Spliting the dataset into test and train set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)

#fitting multiple linear regression into training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set reasult
y_pred=regressor.predict(x_test)

#Checking multiple R2 value
from sklearn.metrics import r2_score
Score=r2_score(y_test,y_pred)

#RMSE value
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse=np.sqrt(mean_squared_error(y_test,y_pred))

#Log transformation (Giving -inf value)
'''startup_l=np.log(startup.iloc[:,:])
x_l=startup_l.iloc[:,0:3]

x_train,x_test,y_train,y_test=train_test_split(x_l,y,test_size=0.4)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

from sklearn.metrics import r2_score
Score_l=r2_score(y_test,y_pred)

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE_l=np.sqrt(mean_squared_error(y_test,y_pred))'''

#Exponential transformation
y_l=np.log(y)

x_train,x_test,y_train,y_test=train_test_split(x,y_l,test_size=0.4,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

from sklearn.metrics import r2_score
Score_ex=r2_score(y_test,y_pred)

from sklearn.metrics import mean_squared_error
from math import sqrt
RMSE_ex=np.sqrt(mean_squared_error(y_test,y_pred))

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
#Plotting imp features
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(x,y)

feach_imp=pd.Series(model.feature_importances_,index=x.columns)
feach_imp.plot(kind='barh')
plt.show()


#Modeling after selection:
x_n=startup.iloc[:,0:3]
y_n=startup.iloc[:,4]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_n,y_n,test_size=0.4,random_state=0)

#fitting multiple linear regression into training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#Predicting the test set reasult
y_pred=regressor.predict(x_test)

#Checking multiple R2 value
from sklearn.metrics import r2_score
Score_f=r2_score(y_test,y_pred)

#RMSE value
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse_f=np.sqrt(mean_squared_error(y_test,y_pred))

x_n.describe()
y_n.describe()






