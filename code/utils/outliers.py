from utils.diagnostics import *
from utils.load_data import *
import numpy as np
import nibabel as nib
import numpy.linalg as npl
import matplotlib.pyplot as plt

# function to remove extended difference outliers 
def remove_outliers(data):
	"""remove outliers based on root mean square differences between volumes 

    Parameters
    ----------
    data : numpy array 
    subject's numpy array indicating fMRI data 

    Returns
    -------
    data_without_outliers:
    numpy arry without the outlying volumes 

    Example
    -------
    >>> data = get_image(1,1).get_data()
    >>> data_without_outliers = remove_outliers(data)
    >>> data_without_outliers.shape 
    (64, 64, 34, 236)
    """
	rms = vol_rms_diff(data)
	outliers_rms = iqr_outliers(rms)[0]
	## append 0 so it is the same legnth as the number of volumes in the data 
	rms.append(0)
	extended_outliers = extend_diff_outliers(outliers_rms)
	data_without_outliers = np.delete(data, extended_outliers,axis=3)
	return data_without_outliers






if __name__ == '__main__':
    import doctest
    doctest.testmod()


