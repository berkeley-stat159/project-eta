import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from utils.load_data import *
import os
from statsmodels.tools.sm_exceptions import PerfectSeparationError



# logistic regresssion fit function 
def apply_logistic(df):
	#removing negative value 
	df = df[df.respcat >= 0]
	#adding intercept column 
	df['intercept'] = 1.0
	train_cols = df.columns[[1,2,7]]
	#logistic regression fit 
	logit = sm.Logit(df['respcat'], df[train_cols])
	# fit the model
	result = logit.fit()
	return result.params 


def apply_logistic_all():
	params = pd.DataFrame()
	for sub in range(1,17):
		print(sub)
		for run in range(1,4):
			print(run)
			df = get_behav(run,sub)
			subject_string = "subject " + str(sub)
			run_string = "run " + str(run)
			results_coeff = apply_logistic(df)
			results_coeff['subject'] = subject_string
			results_coeff['run'] = run_string
			params = params.append(results_coeff, ignore_index = True)
	return params

fit_df = apply_logistic_all()

#output final data frame 
df.to_csv(r'logistic_fit', header=True, index=None, sep=' ', mode='a')


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

