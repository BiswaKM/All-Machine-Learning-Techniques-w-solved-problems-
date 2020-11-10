import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
%matplotlib inline
import seaborn as sns

# Loading data & Checking the shape of data
fraud=pd.read_csv('Fraud_check.csv')
fraud.isnull().sum()
fraud.columns
fraud['Outcome']=np.where(fraud['Taxable.Income']<=30000,'Riski','Good')

#Segregating Numerical & Categorical columns
numerical=fraud.select_dtypes(exclude=object)
categorical=fraud.select_dtypes(include=object)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
fraud['Undergrad']=le.fit_transform(fraud['Undergrad'])
fraud['Marital.Status']=le.fit_transform(fraud['Marital.Status'])
fraud['Urban']=le.fit_transform(fraud['Urban'])

# Pair Plot
sns.pairplot(fraud,hue='Outcome',palette='plasma')
plt.show()

fraud['Outcome']=le.fit_transform(fraud['Outcome'])

# EDA

# Corelation Plot
cor=fraud.corr()
sns.heatmap(cor,annot=True)
plt.show()

# Prob plot
import scipy.stats as stats
import pylab
for i in numerical.columns:
    stats.probplot(numerical[i],dist='norm',plot=pylab)
    plt.show()

# Distribution plots
for i in numerical.columns:
    skew=numerical[i].skew()
    sns.distplot(numerical[i],label='Skewness=%0.2f'%skew)
    plt.legend(loc='best')
    plt.show()

# Segregating input & output data
x=fraud.iloc[:,:-1]
y=fraud.iloc[:,-1]
x=x.drop(['Taxable.Income'],axis=1)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x.iloc[:,2:4]=sc.fit_transform(x.iloc[:,2:4])

# Model Building
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)
model=DecisionTreeClassifier(criterion='entropy')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
score=accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))












