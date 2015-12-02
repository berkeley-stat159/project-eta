from scipy.ndimage import gaussian_filter

def smooth_data(data, SD) :
	"""Smooth data based on gaussian_filter 

	Parameters
	----------
	data: numpy array 
	unsmootheed fMRI data 

	SD:
	the standard deviation for gaussian smoothing 


	Returns
	-------

	nparray of same shape as input but smoothed 

	
	"""
	smoothed_data = gaussian_filter(data, [SD, SD, SD, 0])
	return smoothed_data


