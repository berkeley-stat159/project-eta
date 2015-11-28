import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np


#applied for each run for each subject

df = pd.read_csv("behavdata.txt",delim_whitespace=True)

#removing negative value 

df[df.respcat > 0]

# take a look at the dataset

#adding intercept column 
df['intercept'] = 1.0


print df.heaed()
#variables to fit 

# logistic regresssion fit function 
def apply_logistic(x):
	train_cols = df.columns[[1,2,7]]

	#logistic regression fit 
	logit = sm.Logit(df['respcat'], df[train_cols])

	# fit the model
	result = logit.fit()

	return result



# print results 


print result.summary()

"""
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                respcat   No. Observations:                   86
Model:                          Logit   Df Residuals:                       83
Method:                           MLE   Df Model:                            2
Date:                Fri, 27 Nov 2015   Pseudo R-squ.:                  0.8420
Time:                        15:33:29   Log-Likelihood:                -6.9722
converged:                       True   LL-Null:                       -44.121
                                        LLR p-value:                 7.356e-17
==============================================================================
                 coef    std err          z      P>|z|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
gain           1.0884      0.474      2.296      0.022         0.159     2.018
loss          -1.1493      0.478     -2.404      0.016        -2.086    -0.212
intercept     -2.7703      3.024     -0.916      0.360        -8.697     3.156
==============================================================================
"""

