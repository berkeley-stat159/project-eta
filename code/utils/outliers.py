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
	## append 0 so it is the same legnth as the number of volumes in the data 
	rms.append(0)
	extended_outliers = extend_diff_outliers(outliers_rms)
	data_without_outliers = np.delete(data, extended_outliers,axis=3)
	return data_without_outliers







#Plots

#subject 1 fMRI data 
data = get_image(1,1).get_data()
std = vol_std(data)
plt.plot(std)



outliers = diagnostics.iqr_outliers(std)[0]
ranges = diagnostics.iqr_outliers(std)[1]

plt.axhline(y = ranges[0], linestyle="--", color = "g")
plt.axhline(y = ranges[1], linestyle="--", color = "g")



# RMS outliers plot 

rms = vol_rms_diff(data)



outliers_rms = iqr_outliers(rms)[0]
ranges = iqr_outliers(rms)[1]

outliers_values = []
for i in outliers_rms:
	outliers_values.append(rms[i])


plt.plot(rms)
plt.plot(outliers_rms,outliers_values,"ro", marker = "o")
plt.axhline(y = ranges[0], linestyle="--", color = "g")
plt.axhline(y = ranges[1], linestyle="--", color = "g")


#extended difference outliers 

rms.append(0)

extended_outliers = extend_diff_outliers(outliers_rms)

outliers_values_extended = []
for i in extended_outliers:
	outliers_values_extended.append(rms[i])


plt.plot(rms)
plt.plot(extended_outliers,outliers_values_extended,"ro", marker = "o")

plt.axhline(y = ranges[0], linestyle="--", color = "g")
plt.axhline(y = ranges[1], linestyle="--", color = "g")

if __name__ == '__main__':
    import doctest
    doctest.testmod()


