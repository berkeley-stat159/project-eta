import numpy as np  # the Python array package
import pandas as pd
import matplotlib.pyplot as plt  # the Python plotting package
from scipy.stats import gamma
import nibabel as nib 
import numpy.linalg as npl
from utils.load_data import *
from utils.stimuli import events2neural 




def hrf(times):
  """Produce hemodynamic response function for 'times'

  Parameters
  ----------
  times : times in seconds for the events 

  Returns
  -------

  nparray of length == len(times) with hemodynamic repsonse values for each time course

  Example
  -------
  >>> tr_times = np.arange(240) * 2
  >>> hrf(tr_times)[:5]
  array([ 0.        ,  0.13913511,  0.6       ,  0.58888885,  0.25576589])
  """
  # Gamma pdf for the peak
  peak_values = gamma.pdf(times, 6)
  # Gamma pdf for the undershoot
  undershoot_values = gamma.pdf(times, 12)
  # Combine them
  values = peak_values - 0.35 * undershoot_values
  # Scale max to 0.6
  return values / np.max(values) * 0.6




#build gain and loss columns of the design matrix

def build_design(data,behavdata):
  """Builds design matrix with columns for gain,loss, and a convolved regressor

  Parameters
  ----------
  data : fMRI data for a singe sunject 

  behavdata: behavioral data for a single subject 

  condition: condition to convolved the hemodynamic response function 


  Returns
  -------
  design matrix in the form of a numpy array with 4 columns 

  Example
  -------
  >>> data = get_image(1,1).get_data()
  >>> behavdata = get_behav(1,1)
  >>> build_design(data,behavdata).shape 
  (240,4)
  """ 

  gains = behavdata['gain']
  losses = behavdata['loss']
  TR = 2
  n_vols = data.shape[-1]
  neural_prediction = time_course_behav(behavdata, TR, n_vols)
  tr_times = np.arange(n_vols) * TR
  #buidling gain losses columns of design matrix
  gain_loss = np.zeros((neural_prediction.shape[0], 2))
  j = 0
  for i in range(len(neural_prediction)):
    if neural_prediction[i] != 0:
        gain_loss[i,0] = gains[j]
        gain_loss[i, 1] = losses[j]
        j = j + 1
  #building last column of the design matrix   
  gains = gain_loss[:, 0]
  losses = gain_loss[:, 1]   
  hrf_at_trs = hrf(tr_times)
  convolved1 = np.convolve(neural_prediction, hrf_at_trs)
  convolved2 = np.convolve(gains, hrf_at_trs)
  convolved3 = np.convolve(losses, hrf_at_trs)
  n_to_remove = len(hrf_at_trs) - 1
  convolved1 = convolved1[:-n_to_remove]
  convolved2 = convolved2[:-n_to_remove]
  convolved3 = convolved3[:-n_to_remove]
  #final steps of design
  design = np.ones((len(convolved), 4))
  design[:, 1] = convolved1
  design[:, 2] = convolved2
  design[:, 3] = convolved3
  return design



def regression_fit(data, design): 
  """Finally uses the design matrix from build_design() and fits a linear regression to each voxel 

  Parameters
  ----------
  data : fMRI data for a singe sunject 

  design: matrix returned by build_design()

  Returns
  -------
  numpy array of estimated betas for each voxel

  Example
  -------
  >>> data = get_image(1,1).get_data()
  >>> behavdata = get_behav(1,1)
  >>> design  = build_design(data,behavdata)
  >>> regression_fit(data, design).shape 
  (8, 139264)

  """ 
  data_2d = np.reshape(data, (-1, data.shape[-1]))
  betas_2d = npl.pinv(design).dot(data_2d.T)
  return betas_2d


if __name__ == '__main__':
    import doctest
    doctest.testmod()




