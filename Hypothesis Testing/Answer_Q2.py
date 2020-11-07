import pandas as pd
from scipy import stats
import statsmodels.api as sm

tat=pd.read_csv('LabTAT.csv')
tat.columns

# Normality Test
print(stats.shapiro(tat['Laboratory 1']))
print(stats.shapiro(tat['Laboratory 2']))
print(stats.shapiro(tat['Laboratory 3']))
print(stats.shapiro(tat['Laboratory 4']))

# Variance test
scipy.stats.levene(tat['Laboratory 1'],tat['Laboratory 2'])
scipy.stats.levene(tat['Laboratory 2'],tat['Laboratory 3'])
scipy.stats.levene(tat['Laboratory 3'],tat['Laboratory 4'])
scipy.stats.levene(tat['Laboratory 1'],tat['Laboratory 4'])

#Anova
from statsmodels.formula.api import ols
tat.columns='lab1','lab2','lab3','lab4'
mod = ols('lab1 ~ lab2+lab3+lab4',data=tat).fit()

aov_table=sm.stats.anova_lm(mod, type=2)
print(aov_table)
