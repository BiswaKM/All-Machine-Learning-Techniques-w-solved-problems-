import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

df=pd.read_csv('Diabetes.csv')

df.shape
df.info()
df.isnull().sum()
a=df.describe().T
df.columns

# EDA
plt.figure()
sns.pairplot(df,hue=' Class variable',palette='plasma')
plt.show()

plt.figure()
corr=df.corr()
sns.heatmap(corr,annot=True)

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df[' Class variable']=le.fit_transform(df[' Class variable'])

plt.figure()
for i in df.columns:
    for j in df.columns:
        sns.jointplot(x=i,y=j,data=df,kind='kde',color='k')
        plt.show()

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df.iloc[:,0:8]=sc.fit_transform(df.iloc[:,0:8])

#Segregating Feature & target column
x=df.iloc[:,0:8]
y=df.iloc[:,-1]

# Modeling
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
model=RandomForestClassifier(n_estimators=10)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

score=accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
print(classification_report(y_test,y_pred))

#Feature Selection
from sklearn.ensemble import ExtraTreesClassifier
classifier=ExtraTreesClassifier()
classifier.fit(x,y)
importance=pd.Series(classifier.feature_importances_,index=x.columns)
importance_score=pd.DataFrame(importance).reset_index()

importance_score.plot(kind='barh')
x_f=x.drop(columns=[' Number of times pregnant',' Diastolic blood pressure', ' Triceps skin fold thickness',
       ' 2-Hour serum insulin',' Diabetes pedigree function'])
y_f=y

xf_train,xf_test,yf_train,yf_test=train_test_split(x_f,y_f,test_size=0.3,random_state=0)
model=RandomForestClassifier(n_estimators=10)
model.fit(xf_train,yf_train)
yf_pred=model.predict(xf_test)

score_f=accuracy_score(yf_test,yf_pred)
confusion_matrix(yf_test,yf_pred)
print(classification_report(yf_test,yf_pred))






