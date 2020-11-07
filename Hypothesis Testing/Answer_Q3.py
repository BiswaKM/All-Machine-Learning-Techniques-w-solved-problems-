import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm
import numpy as np

sales=pd.read_csv('BuyerRatio.csv')

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
sales['Observed Values']=le.fit_transform(sales['Observed Values'])

Chisquares_results=scipy.stats.chi2_contingency(sales)

Chi_square=[['','Test Statistic','p-value'],['Sample Data',Chisquares_results[0],Chisquares_results[1]]]
print(Chi_square)
