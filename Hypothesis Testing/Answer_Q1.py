import pandas as pd
import scipy 
from scipy import stats
import statsmodels.api as sm

cutlets=pd.read_csv('Cutlets.csv')
cutlets.columns
# Normality Test
print(stats.shapiro(cutlets['Unit A']))
print(stats.shapiro(cutlets['Unit B']))

# Variance test
scipy.stats.levene(cutlets['Unit A'], cutlets['Unit B'])

# Sample T test
scipy.stats.ttest_ind(cutlets['Unit A'],cutlets['Unit B'])
