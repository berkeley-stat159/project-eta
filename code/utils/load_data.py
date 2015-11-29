from __future__ import print_function, division

import random

import numpy as np
import pandas as pd
import nibabel as nib

from nums import n_convert


def get_image(r, s):
    """Load .nii file for subject `s` on run `r`

    Parameters
    ----------
    r : int
        Run number
    s : int
        Subject number

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

def get_cond(r, s, c):
    """Load condition data file `c` for subject `s` on run `r`

    Parameters
    ----------
    r : int
        Run number
    s : int
        Subject number
    c : int
        Condition file

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

def get_behav(r, s):
    """Load behavioral data file for subject `s` on run `r`
    Remove bad activations based on `respcat`

    Parameters
    ----------
    s : int
        Subject number
    r : int
        Run number

    Returns
    -------
    behav : pd.DataFrame

    Examples
    --------
    >>> get_behav(1, 1)[:5]
       onset  duration  gain  loss  PTval  respnum  respcat     RT
    0      4         3    18    12   6.12        2        1  1.793
    1      8         3    10    15  -4.85        3        0  1.637
    2     18         3    34    16  18.16        1        1  1.316
    3     24         3    18     5  13.05        1        1  1.670
    4     28         3    26    13  13.13        2        1  1.232
    """
    s, r = n_convert(s, r)
    f_behav = '../data/ds005/sub'+s+\
              '/behav/task001_run'+r+\
              '/behavdata.txt'
    behav = pd.read_csv(f_behav, sep='\t')
    behav.insert(1, 'duration', 3)
    behav = behav[behav.respcat>=0]
    behav.reset_index(drop=True, inplace=True)
    return behav

def n_load(fn, args, kwargs=None):
    """Load an arbitrary number of objects

    Parameters
    ----------
    fn : function
        The function to use for loading data
    args : tuple or list
        Arguments of variable length
    kwargs : dict
        Keyword arguments of variable length

    Returns
    -------
    n_loaded: tuple
        Tuple of length *args

    Examples
    --------
    >>> n_load(get_behav, [1, 2], {'s' : 1})[0][:5]
       onset  duration  gain  loss  PTval  respnum  respcat     RT
    0      4         3    18    12   6.12        2        1  1.793
    1      8         3    10    15  -4.85        3        0  1.637
    2     18         3    34    16  18.16        1        1  1.316
    3     24         3    18     5  13.05        1        1  1.670
    4     28         3    26    13  13.13        2        1  1.232
    >>> n_load(get_behav, [1, 2], {'s' : 1})[1][:5]
       onset  duration  gain  loss  PTval  respnum  respcat     RT
    0      0         3    20     5  15.05        1        1  1.290
    1      4         3    22    17   5.17        2        1  1.163
    2      8         3    10    16  -5.84        4        0  1.265
    3     12         3    38    18  20.18        2        1  1.488
    4     16         3    20    14   6.14        2        1  1.446
    >>> bh1, bh2, bh3 = n_load(get_behav, [1, 2, 3], {'s' : 1})
    >>> n_load(time_course_behav, [bh1, bh2, bh3], {'TR':2, 'n_trs':240})[0]
    array([ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  1.,
            0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,
            1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  1.,  0.,
            1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,
            0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,
            1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,
            1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,
            1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,
            1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,
            1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,
            1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,
            0.,  0.,  0.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,
            0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  0.,  1.,
            0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,
            1.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,
            0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  0.,  1.,  0.,  1.,  0.,
            1.,  0.,  1.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  1.,  0.,  1.,
            0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.,  1.,
            0.,  1.,  0.,  1.,  0.,  0.])
    """
    if kwargs is None:
        n_loaded = tuple([fn(r) for r in args])
    else:
        n_loaded = tuple([fn(r, **kwargs) for r in args])
    if len(n_loaded) == 1:
        return n_loaded[0]
    else:
        return n_loaded

def time_course_behav(behav, TR, n_trs):
    """Use `behavdata.txt` to create the neural time course

    Parameters
    ----------
    behav : pd.DataFrame
        The data returned from `get_behav()`
    TR : int
        Repetition Time in seconds
    n_trs : int
        The number of repetitions in a functional run

    Returns
    -------
    time_course : np.ndarray
        The neural time course based on `behavdata.txt`

    Examples
    --------
    >>> b = get_behav(1, 1)
    >>> time_course_behav(b, 2, 240)[:10]
    array([ 0.,  0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.,  1.])
    """
    ons_durs = np.floor(behav.iloc[:,:2].values / TR).astype(int)
    time_course = np.zeros(n_trs)
    for onset, duration in ons_durs:
        time_course[onset : onset+duration] = 1
    return time_course


if __name__ == '__main__':
    import doctest
    doctest.testmod()
