import numpy as np 
from utils.linear_fit import *
from utils.outliers import remove_outliers
from utils.load_data import get_behav
from utils.smoothing import *
import pandas as pd 
import pylab as pl 
import os
from os.path import join, getsize

def apply_ols_to_subject(total_s, total_r, r_outliers = False, smooth = False):	
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
	>>> betas_2d = apply_ols_to_subject(total_s, total_r)
	>>> betas_2d.shape
	(96, 139264)
	
	"""
	#for sub in range(total_s+1)[1:]:
		#for run in range(total_r+1)[1:]:
	for sub in range(1,17):
		for run in range(1,4):
			data = get_image(run, sub).get_data()
			if r_outliers == True:
				data = remove_outliers(data)
			if smooth == True:
				data = smooth_data(data, 2)
			behavdata = get_behav(run, sub)
			print("run:", run, "sub:", sub)
			design = build_design(data, behavdata)
			if sub == 1 and run == 1:
				gain_loss_betas_2d = regression_fit(data, design)[2:,:]
			else: 
				betas = regression_fit(data, design)[2:,:]
				gain_loss_betas_2d = np.concatenate((gain_loss_betas_2d, betas), axis=0)
	
	return gain_loss_betas_2d

def average_betas(betas_2d):
	"""Averaging betas over all subjects and runs

	Parameters
	----------
	betas_2d : a m x n numpy array, where m is the total number of runs for all subjects
	and n is the total number of voxels

	Returns
	-------
	A 2 x n numpy array, containing the averaged beta(gain) and beta(loss) for each voxel

	Example
	-------
	>>> total_s = 16
	>>> total_r = 3
	>>> betas_2d = apply_ols_to_subject(total_s, total_r)
	>>> betas_avg = average_betas(betas_2d)
	>>> betas_avg.shape
	(2, 139264)

	"""
	gain_average = np.mean(betas_2d[::2], axis=0)
	loss_average = np.mean(betas_2d[1::2], axis=0)
	return np.array((gain_average, loss_average))

def beta_2d_to_4d(betas_2d):
    """Convert the model coefficients from 2-dimensional to 4-dimensional

    Parameters
    ----------
    betas_2d : an m x n numpy array, where m is the total number of runs for all subjects
	and n is the total number of voxels

	Returns
	-------
	An A x B x C x D matrix, where A x B x C is the same as the first 3 dimensions
	of the fMRI data and D is the number of coefficients in the model.

	Example
	-------
	>>> total_s = 16
	>>> total_r = 3
	>>> betas_2d = apply_ols_to_subject(total_s, total_r)
	>>> betas_4d = beta_2d_to_4d(betas_2d)
	>>> betas_4d.shape
	(64, 64, 34, 4) 

    """
    betas_4d = np.reshape(betas_2d.T, (64,64,34) + (-1,))
    return betas_4d

def betas_middle_slice_graph(betas_4d):
	"""Plot betas for the middle slice of brain

	Parameters
	----------
	betas_4d : an A x B x C x D matrix, where A x B x C is the same as the first 3 dimensions
	of the fMRI data and D is the number of coefficients in the model.

	Returns
	-------
	Saved plots of averaged coefficients (gain and loss) for the middle slice of the brain
	"""
	plt.imshow(betas_4d[:, :, 16, 0], interpolation='nearest', cmap='gray')
	plt.title('Middle Slice Beta(Gain)')
	plt.colorbar()
	plt.savefig('middle_slice_gain.png')
	plt.close()
	plt.imshow(betas_4d[:, :, 16, 1], interpolation='nearest', cmap='gray')
	plt.title('Middle Slice Beta(Loss)')
	plt.colorbar()
	plt.savefig('middle_slice_loss.png')
	plt.close()


if __name__ == '__main__':
    import doctest
    doctest.testmod()




