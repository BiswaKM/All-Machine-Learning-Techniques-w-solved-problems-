import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# 1 - Data Loding
zoo=pd. read_csv('C:\\Users\\Biswa\\Desktop\\360DigiTMG\\18 . Module 18 - Machine Learning K-Nearest Neighbour\\Assignment\\Dataset\\zoo.csv')

zoo.head()
zoo.shape
zoo.columns
zoo.info()
zoo.isnull().sum()
zoo.describe()

# 2- EDA

# Type wise animal catSegory plots
zoo_eda=zoo.drop('animal name',axis=1)

for i in zoo_eda.columns:
    fig=plt.subplots(figsize=(12,6))
    sns.countplot(zoo_eda['type'],hue=zoo_eda[i])
    plt.legend(loc='upper right')
    plt.title(i)
    plt.show()

for i in zoo_eda.columns:
    g=sns.catplot(i,col='type',data=zoo_eda,kind='count',col_wrap=7,height=2.5,aspect=0.8)
    plt.show()
    
 # Corelation Heatmap
cor=zoo.corr()  
figure=plt.subplots(figsize=(12,8))
sns.heatmap(cor,xticklabels='auto',yticklabels='auto',cmap='coolwarm',annot=True)
plt.show()

#Segegating features & target columns

x=zoo.iloc[:,1:-1]
y=zoo.iloc[:,-1]

# Vif Values
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif=[variance_inflation_factor(zoo_eda.values,i) for i in range(zoo_eda.shape[1])]
vif=pd.DataFrame({'VIF':vif},index=zoo_eda.columns)
vif=vif.reset_index()

vif_score=sns.catplot(x='VIF',y='index',kind='bar',data=vif,aspect=2)
(vif_score.set_titles('Vif Score'))

# Feature Importance
from sklearn.ensemble import ExtraTreesClassifier

model=ExtraTreesClassifier()
model.fit(x,y)
feature_importance=pd.Series(model.feature_importances_,index=x.columns)
feature_importance=pd.DataFrame(feature_importance)
feature_importance.columns=['Importance']
feature_importance=feature_importance.reset_index()

feature_score=sns.catplot(y='index',x='Importance',data=feature_importance,kind='bar',aspect=2)
plt.show()

x_new=zoo_eda.drop(columns=['predator','venomous','domestic','catsize'])

# Modeling:
    
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

accuracy_score=[]
for i in range(1,40,2):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x_new,y,cv=10)
    accuracy_score.append(score.mean())    

f,ax=plt.subplots(figsize=(12,6)) 
plt.plot(range(1,40,2),accuracy_score,marker='o')
plt.xlabel('K Number')
plt.ylabel('Accuracy')
plt.title('Accuracy rate')
plt.show()

# Modeling with k=3

knn1=KNeighborsClassifier(n_neighbors=3)
score1=cross_val_score(knn1,x_new,y,cv=10)

score1.mean()
score1.max()
score1.min()

# Confusion matrix & Classificatoin report
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.metrics import confusion_matrix,classification_report

knn2=KNeighborsClassifier(n_neighbors=3)
knn2.fit(x_train,y_train)

y_pred=knn2.predict(x_test)

cm=confusion_matrix(y_test,y_pred)
report=classification_report(y_test,y_pred)
print(report)





