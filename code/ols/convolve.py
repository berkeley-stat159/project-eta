# Author: Wenjie Xu

import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
from scipy.stats import gamma
import nibabel as nib 
import numpy.linalg as npl
img = nib.load('bold.nii')
data = img.get_data()
data = data[...,1:]

### 1. Creating the design matrix  

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

def gain_loss_x(gains, losses, neural_prediction):
    x = np.zeros((neural_prediction.shape[0], 2))
    j = 0
    for i in range(len(neural_prediction)):
        if neural_prediction[i] != 0:
            x[i,0] = gains[j]
            x[i, 1] = losses[j]
            j = j + 1
    return x

TR = 2
n_vols = data.shape[-1]
from stimuli import events2neural 
neural_prediction = events2neural('cond001.txt', TR, n_vols)
tr_times = np.arange(n_vols) * TR

hrf_at_trs = hrf(tr_times)
convolved = np.convolve(neural_prediction, hrf_at_trs)
n_to_remove = len(hrf_at_trs) - 1
convolved = convolved[:-n_to_remove]

# plot the convolved data
plt.plot(tr_times, neural_prediction)
plt.plot(tr_times, convolved)
plt.show()
# the plot scale is too large, better to just look at a part of it

names = get_behave('behavdata.txt')[0]
# names = ['onset', 'gain', 'loss', 'PTval', 'respnum', 'respcat', 'RT']
task = get_behave('behavdata.txt')[1:]
gains = np.array([line[1] for line in task], dtype = int)
losses = np.array([line[2] for line in task], dtype = int)
x = gain_loss_x(gains, losses, neural_prediction)

design = np.ones((len(convolved), 4))
design[:, 1] = convolved
design[:, 2:] = x

### 2. Fitting an OLS model 
## a. Before orthogonalizing 
import statsmodels.api as sm

def ols_betas(data, design):
    data_2d = np.reshape(data, (-1, data.shape[-1]))
    df = design.shape[0] - npl.matrix_rank(design)
    for i in range(data_2d.T.shape[-1]):
        y = data_2d.T[:, i]
        model = sm.OLS(y, design)
        results = model.fit()
        if i == 0:
            betas = results.params
            r_sq = results.rsquared
            residuals = y - results.predict()
            MRSS = np.sum(residuals ** 2, axis = 0)/df
        else: 
            betas = np.vstack((betas, results.params))
            r_sq = np.append(r_sq, results.rsquared)
            residuals = np.vstack((residuals, y - results.predict()))
            MRSS = np.vstack((MRSS, np.sum(residuals ** 2, axis = 0)/df))
    return 


betas = npl.pinv(design).dot(data_2d.T)
y_hat = design.dot(betas) 
residuals = data_2d.T - y_hat
RSS = np.sum(residuals ** 2, axis = 0)
df = design.shape[0] - npl.matrix_rank(design)
MRSS = RSS/df
meanMRSS1 = np.mean(MRSS)
# 33.547787836605082

betas_4d = np.reshape(betas.T, img.shape[:-1] + (-1,))
plt.imshow(betas_4d[:, :, 16, 1], interpolation='nearest', cmap='gray') # middle slice for beta1
plt.imshow(betas_4d[:, :, 16, 0], interpolation='nearest', cmap='gray') # middle slice for beta0

# Seems middle and upper (30) parts have higher beta1s. 


