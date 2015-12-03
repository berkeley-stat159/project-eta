import numpy as np 
from utils.linear_fit import *
import pandas as pd 
import pylab as pl 
import os
from os.path import join, getsize

def apply_ols_to_subject(total_s, total_r):	
	"""Apply OLS model to all runs of all subjects in the ds005 data folder

	Parameters
	----------
	total_s : total number of subjects

	total_r : total number of runs for each subjects


	Returns
	-------
	Beta(loss) and Beta(gain) for each voxel stacked over all runs of all subjects
	
	Example 
	-------
	>>> total_s = 16
	>>> total_r = 3
	>>> 
	"""
	for sub in range(total_s+1)[1:]:
		for run in range(total_r+1)[1:]:
			data = get_image(run, sub).get_data()
			behave = get_behav(run, sub)
			design = build_design(data, behavdata)
			if sub == 1 and run == 1:
				gain_loss_betas_2d = regression_fit(data, design)[2:,:]
			else: 
				betas = regression_fit(data, design)[2:,:]
				gain_loss_betas_2d = np.concatenate((gain_loss_betas_2d, betas), axis=0)
	return gain_loss_betas_2d

def average_betas(betas):

	rootdir = '../data/ds005'
	sub_lst = ['sub00' + str(i) for i in range(17)[1:]]
	run_lst = ['task001_run00' + str(1) for i in range(4)[1:]]
	for sub in sub_lst: 
		for run in run_lst:
			behave = pd.read_csv(join(rootdir, sub, 'behav', run, "behavdata.txt"), delim_whitespace=True)



