import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm
import numpy as np

cof=pd.read_csv('CustomerOrderform.csv')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
cof['Phillippines']=le.fit_transform(cof['Phillippines'])
cof['Indonesia']=le.fit_transform(cof['Indonesia'])
cof['Malta']=le.fit_transform(cof['Malta'])
cof['India']=le.fit_transform(cof['India'])
cof.columns

count=cof['Phillippines'].value_counts()
count=pd.DataFrame(count)
count['Indonesia']=cof['Indonesia'].value_counts()
count['Malta']=cof['Malta'].value_counts()
count['India']=cof['India'].value_counts()

Chisquares_results=scipy.stats.chi2_contingency(count)

Chi_square=[['','Test Statistic','p-value'],['Sample Data',Chisquares_results[0],Chisquares_results[1]]]
print(Chi_square)
