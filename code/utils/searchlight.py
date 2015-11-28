from __future__ import print_function, division

from itertools import product

import numpy as np


def inrange(arr, b, f, ind):
    """Ensuring slices are possible.

    Parameters
    ----------
    arr: np.ndarray
        2d or 3d input array
    b: int
        The backward adjusted value
    f: int
        The forward adjusted value
    ind: int
        The index of `arr` to consider

    Returns:
    b, f: tuple
        The corrected slice values
    """
    d_min = 0
    d_max = arr.shape[ind] - 1
    if b < d_min:
        b = 0
    if f > d_max:
        f = d_max
    return b, f

def sphere(arr, c, r=1):
    """Selecting adjacent voxels in 3d space, across time
    
    Parameters
    ----------
    arr: np.ndarray
        4d input array
    c: tuple (x, y, z)
        The center voxel from which the volume is created
    r: int
        The radius for the sphere
        
    Returns
    -------
    adjacent: np.ndarray
        A sphere for every time time slice

    Examples
    --------
    >>> X = np.arange(54).reshape(3, 3, 3, 2)
    >>> sphere(X, (1, 1, 1))
    array([[[[ 0,  1],
             [ 2,  3],
             [ 4,  5]],
    <BLANKLINE>
            [[ 6,  7],
             [ 8,  9],
             [10, 11]],
    <BLANKLINE>
            [[12, 13],
             [14, 15],
             [16, 17]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[18, 19],
             [20, 21],
             [22, 23]],
    <BLANKLINE>
            [[24, 25],
             [26, 27],
             [28, 29]],
    <BLANKLINE>
            [[30, 31],
             [32, 33],
             [34, 35]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[36, 37],
             [38, 39],
             [40, 41]],
    <BLANKLINE>
            [[42, 43],
             [44, 45],
             [46, 47]],
    <BLANKLINE>
            [[48, 49],
             [50, 51],
             [52, 53]]]])
    >>> Y = np.arange(5**4).reshape(5, 5, 5, 5)
    >>> sphere(Y, (2, 2, 2))
    array([[[[155, 156, 157, 158, 159],
             [160, 161, 162, 163, 164],
             [165, 166, 167, 168, 169]],
    <BLANKLINE>
            [[180, 181, 182, 183, 184],
             [185, 186, 187, 188, 189],
             [190, 191, 192, 193, 194]],
    <BLANKLINE>
            [[205, 206, 207, 208, 209],
             [210, 211, 212, 213, 214],
             [215, 216, 217, 218, 219]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[280, 281, 282, 283, 284],
             [285, 286, 287, 288, 289],
             [290, 291, 292, 293, 294]],
    <BLANKLINE>
            [[305, 306, 307, 308, 309],
             [310, 311, 312, 313, 314],
             [315, 316, 317, 318, 319]],
    <BLANKLINE>
            [[330, 331, 332, 333, 334],
             [335, 336, 337, 338, 339],
             [340, 341, 342, 343, 344]]],
    <BLANKLINE>
    <BLANKLINE>
           [[[405, 406, 407, 408, 409],
             [410, 411, 412, 413, 414],
             [415, 416, 417, 418, 419]],
    <BLANKLINE>
            [[430, 431, 432, 433, 434],
             [435, 436, 437, 438, 439],
             [440, 441, 442, 443, 444]],
    <BLANKLINE>
            [[455, 456, 457, 458, 459],
             [460, 461, 462, 463, 464],
             [465, 466, 467, 468, 469]]]])
    >>> (sphere(X, (1, 1, 1)) == X).all()
    True
    >>> (sphere(Y, (2, 2, 2), 2) == Y).all()
    True
    """
    
    # checking type
    assert isinstance(arr, np.ndarray), 'input array must be type np.ndarray'
    assert isinstance(c, tuple), 'center must be type tuple'
    assert isinstance(r, int), 'radius must be type int'
        
    # checking size
    assert len(arr.shape) == 4, 'array must be 4-dimensional'
    assert len(c) == 3, 'tuple length must be 3'
    assert len(c) == len(arr.shape)-1, 'tuple length must equal array dim'
    assert r < min(arr.shape[:-1]), 'radius must be smaller than the '+\
        'smallest dimension of the first three axes'

    # checking valid `c` for `arr`; IndexError raised if not
    arr[c]

    b = lambda v, r: v - r
    f = lambda v, r: v + r

    # indices
    x, y, z = c[0], c[1], c[2]

    # reach
    xb, xf = b(x, r), f(x, r)
    yb, yf = b(y, r), f(y, r)
    zb, zf = b(z, r), f(z, r)

    # checking within dimensions
    xb, xf = inrange(arr, xb, xf, 0)
    yb, yf = inrange(arr, yb, yf, 1)
    zb, zf = inrange(arr, zb, zf, 2)
    return arr[xb:xf+1, yb:yf+1, zb:zf+1]

def nonzero_indices(arr):
    """Get the 3d indices for `arr` (4d) where the value is nonzero

    Parameters
    ----------
    arr : np.ndarray
        The image data returned from calling `nib.load(f).get_data()`

    Returns
    -------
    nonzeros : dict
        A dict where the keys are int and the values are tuples

    Examples
    --------
    >>> np.random.seed(42)
    >>> X = np.random.randn(16).reshape(2, 2, 2, 2)
    >>> Y = np.ones((2, 2, 2, 2))
    >>> Z = np.round(X - Y).astype(int)
    >>> nonzero_indices(Z)[0]
    (0, 1, 1)
    >>> nonzero_indices(Z)[4]
    (1, 0, 1)
    """
    nonzero_indices = list(set(zip(*np.nonzero(arr)[:-1])))
    nonzeros = {k : v
                for (k, v) in 
                    zip(range(len(nonzero_indices)), nonzero_indices)}
    return nonzeros


if __name__ == '__main__':
    import doctest
    doctest.testmod()
