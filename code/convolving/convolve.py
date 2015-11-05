# Creating convolved

import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
from scipy.stats import gamma
import nibabel as nib 
img = nib.load('bold.nii')
data = img.get_data()
data = data[...,1:]


def hrf(times):     
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

def get_behave(filename):
	# Read in behave.txt to get repetition times
	df = [line.strip().split('\t') for line in open(filename)]
	return df

names = get_task('behavdata.txt')[0]
# names = ['onset', 'gain', 'loss', 'PTval', 'respnum', 'respcat', 'RT']
behave = get_behave('behavdata.txt')[1:]

tr_times = [float(p[6]) for p in behave] # list of repetition times
n_vol = data.shape[3] # number of volumes 

from stimuli_v1 import events2neural # a new version of stimuli that deals with nonconstant TR
neural_prediction = events2neural('cond001.txt', tr_times, n_vol) 

hrf_at_trs = hrf(tr_times)
convolved = np.convolve(neural_prediction, hrf_at_trs)
n_to_remove = len(hrf_at_trs) - 1
convolved = convolved[:-n_to_remove]


