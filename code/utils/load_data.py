from __future__ import print_function, division

import random

import numpy as np
import pandas as pd
import nibabel as nib

from nums import n_convert


def get_image(s, r):
    """Load .nii file for subject `s` on run `r`

    Parameters
    ----------
    s : int
        subject number
    r : int
        run number

    Returns
    -------
    Nifti1Image object

    Example
    -------
    >>> get_image(1, 1).shape
    (64, 64, 34, 240)
    """
    s, r = n_convert(s, r)
    f_img = '../data/ds005/sub'+s+\
            '/BOLD/task001_run'+r+\
            '/bold.nii.gz'
    return nib.load(f_img)

def get_cond(s, r, c):
    """Load condition data file `c` for subject `s` on run `r`

    Parameters
    ----------
    s : int
        subject number
    r : int
        run number
    c : int
        condition file

    Returns
    -------
    cond : pd.DataFrame
    Examples
    --------
    >>> get_cond(1, 1, 1)[:5]
       onset  duration  amplitude
    0      0         3          1
    1      4         3          1
    2      8         3          1
    3     18         3          1
    4     24         3          1
    """
    s, r, c = n_convert(s, r, c)
    f_cond = '../data/ds005/sub'+s+\
             '/model/model001/onsets/task001_run'+r+\
             '/cond'+c+'.txt'
    cond = pd.DataFrame(np.loadtxt(f_cond),
                        columns=['onset', 'duration', 'amplitude'])
    return cond

def get_behav(s, r):
    """Load behavioral data file for subject `s` on run `r`

    Parameters
    ----------
    s : int
        subject number
    r : int
        run number

    Returns
    -------
    behav : pd.DataFrame
    Examples
    --------
    >>> get_behav(1, 1)[:5]
       onset  duration  gain  loss  PTval  respnum  respcat     RT
    0      0         3    20    15   5.15        0       -1  0.000
    1      4         3    18    12   6.12        2        1  1.793
    2      8         3    10    15  -4.85        3        0  1.637
    3     18         3    34    16  18.16        1        1  1.316
    4     24         3    18     5  13.05        1        1  1.670
    """
    s, r = n_convert(s, r)
    f_behav = '../data/ds005/sub'+s+\
              '/behav/task001_run'+r+\
              '/behavdata.txt'
    behav = pd.read_csv(f_behav, sep='\t')
    behav.insert(1, 'duration', 3)
    return behav


if __name__ == '__main__':
    import doctest
    doctest.testmod()
