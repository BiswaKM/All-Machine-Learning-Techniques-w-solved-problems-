import pandas as pd
import scipy
from scipy import stats
import statsmodels.api as sm
import numpy as np
sales=pd.read_csv('Fantaloons.csv')
from statsmodels.stats.proportion import proportions_ztest
tap1=sales.Weekdays.value_counts()
tap1
tap2=sales.Weekend.value_counts()
tap2
stats,pval = proportions_ztest(tap1, tap2,alternative='two-sided') 
print(pval) 
stats,pval = proportions_ztest(tap1, tap2,alternative='larger')
print(pval)    
