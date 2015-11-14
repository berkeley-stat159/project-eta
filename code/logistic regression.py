import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np



df = pd.read_csv("behavdata.txt",delim_whitespace=True)

#removing negative value 
df['respcat'][0] = 0

# take a look at the dataset
print df.head()


#variables to fit 

train_cols = df.columns[1:3]

#logistic regression fit 
logit = sm.Logit(df['respcat'], df[train_cols])

# fit the model
result = logit.fit()

# print results 


#print result.summary()
#==============================================================================
#Dep. Variable:                respcat   No. Observations:                   86
#Model:                          Logit   Df Residuals:                       84
#Method:                           MLE   Df Model:                            1
#Date:                Wed, 11 Nov 2015   Pseudo R-squ.:                  0.8316
#Time:                        16:52:00   Log-Likelihood:                -7.4308
#converged:                       True   LL-Null:                       -44.121
#                                        LLR p-value:                 1.070e-17
#==============================================================================
#                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
#------------------------------------------------------------------------------
#gain           0.8843      0.325      2.720      0.007         0.247     1.521##
#loss          -1.0894      0.408     -2.671      0.008        -1.889    -0.290
#==============================================================================
#"""

