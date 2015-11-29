import numpy as np
import nibabel as nib
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from utils.load_data import *
from utils.searchlight import sphere, nonzero_indices


def shape_X(X, n_trs, tc):
    """Helper function for shaping X, the 'sphere' from calling
    `searchlight.sphere`. The input shape is (..., n_trs).
    Flatten across the first three dimensions and transpose the matrix.
    Keep only the 'rows' where the time course `tc` is equal to 1.

    Parameters
    ----------
    X : np.ndarray
        The 'sphere'
    n_trs : int
        The number of repetitions in a functional run
    tc : np.ndarray
        The neural time course

    Returns
    -------
    X_ : np.ndarray
        The reshaped array

    Examples
    --------
    >>> Z = np.arange(32).reshape(2, 2, 2, 4)
    >>> n_trs = 4
    >>> tc = np.array([1, 1, 0, 1])
    >>> shape_X(Z, n_trs, tc)
    array([[ 0,  4,  8, 12, 16, 20, 24, 28],
           [ 1,  5,  9, 13, 17, 21, 25, 29],
           [ 3,  7, 11, 15, 19, 23, 27, 31]])
    """
    X_ = X.reshape(-1, n_trs).T
    return X_[tc==1, :]

def make_X(Xi, Xj):
    """Create the training data based on 2d arrays from two runs. Arrays must
    have the same number of columns, but not necessarily rows.

    Parameters
    ----------
    Xi : np.ndarray
        The data (2d) from run `i`
    Xj : np.ndarray
        The data (2d) from run `j`

    Returns
    -------
    X : np.ndarray
        An array of shape (Xi.shape[0]+Xj.shape[0], Xi.shape[1]+Xj.shape[1])

    Examples
    --------
    >>> A = np.arange(8).reshape(4, 2)
    >>> B = np.array([2, 4, 6, 8, 10, 12, 14, 16]).reshape(4, 2)
    >>> make_X(A, B)
    array([[ 0,  1],
           [ 2,  3],
           [ 4,  5],
           [ 6,  7],
           [ 2,  4],
           [ 6,  8],
           [10, 12],
           [14, 16]])
    """
    X = np.vstack((Xi, Xj))
    return X

def make_y(yi, yj):
    """Create the training labels based on two 1d arrays whose lengths
    can be different

    Parameters
    ----------
    yi : np.ndarray
        The 1d labels for run `i`
    yj : np.ndarray
        The 1d labels for run `j`

    Returns
    -------
    y : np.ndarray
        The array of shape (len(yi)+len(yj),)

    Examples
    --------
    >>> a = np.array([1, 1, 0, 1, 0, 1])
    >>> b = np.array([0, 0, 1, 1])
    >>> make_y(a, b)
    array([1, 1, 0, 1, 0, 1, 0, 0, 1, 1])
    """
    y = np.append(yi, yj)
    return y

def classification(s, TR, clf):
    """Classification using `clf` of 'active' states for subject `s`
    across all runs. Using Leave-Block-Out cross-validation (LBO-CV).

    Parameters
    ----------
    s : int
        Subject number
    TR : int
        Repetition Time in seconds
    clf : abc.ABCMeta
        Scikit-Learn classifier, e.g., `sklearn.linear_model.LinearRegression`

    Returns
    -------
    _ : pd.DataFrame
        Voxel index values and classification accuracy scores for each run
    """
    img1, img2, img3 = n_load(get_image, [1, 2, 3], {'s' : s})
    data1, data2, data3 = tuple([i.get_data() for i in [img1, img2, img2]])
    bh1, bh2, bh3 = n_load(get_behav, [1, 2, 3], {'s' : s})

    n_trs = img1.shape[-1]
    tc1, tc2, tc3 = n_load(time_course_behav,
                           [bh1, bh2, bh3],
                           {'TR' : TR, 'n_trs' : n_trs})

    y1 = np.array(bh1.respcat)
    y2 = np.array(bh2.respcat)
    y3 = np.array(bh3.respcat)

    # TODO make this faster in `searchlight.nonzero_indices`
    nz1, nz2, nz3 = n_load(nonzero_indices, [data1, data2, data3])
    nz = list(set(nz1 + nz2 + nz3))
    nonzeros = {k : v for (k, v) in zip(range(len(nz)), nz)}

    for i in nonzeros.keys()[:1]: # TODO remove slice
        X1, X2, X3 = n_load(sphere, [data1, data2, data3], {'c' : nonzeros[i]})
        X1 = shape_X(X1, n_trs, tc1)
        X2 = shape_X(X2, n_trs, tc2)
        X3 = shape_X(X3, n_trs, tc3)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
