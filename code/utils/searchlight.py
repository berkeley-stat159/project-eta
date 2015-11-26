from __future__ import print_function, division

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
    """Selecting adjacent voxels in 2d or 3d space
    
    Parameters
    ----------
    arr: np.ndarray
        2d or 3d input array
    c: tuple ([z,] x, y)
        The center voxel from which the volume is created
    r: int
        The radius of the volume
        
    Returns
    -------
    adjacent: np.ndarray

    Examples
    --------
    >>> X = np.arange(125).reshape(5, 5, 5)
    >>> sphere(X, (2, 2, 2), 1)
    array([[[31, 32, 33],
            [36, 37, 38],
            [41, 42, 43]],
    <BLANKLINE>
           [[56, 57, 58],
            [61, 62, 63],
            [66, 67, 68]],
    <BLANKLINE>          
           [[81, 82, 83],
            [86, 87, 88],
            [91, 92, 93]]])

    >>> sphere(X, (2, 2, 2), 2)
    array([[[  0,   1,   2,   3,   4],
            [  5,   6,   7,   8,   9],
            [ 10,  11,  12,  13,  14],
            [ 15,  16,  17,  18,  19],
            [ 20,  21,  22,  23,  24]],
    <BLANKLINE>
           [[ 25,  26,  27,  28,  29],
            [ 30,  31,  32,  33,  34],
            [ 35,  36,  37,  38,  39],
            [ 40,  41,  42,  43,  44],
            [ 45,  46,  47,  48,  49]],
    <BLANKLINE>
           [[ 50,  51,  52,  53,  54],
            [ 55,  56,  57,  58,  59],
            [ 60,  61,  62,  63,  64],
            [ 65,  66,  67,  68,  69],
            [ 70,  71,  72,  73,  74]],
    <BLANKLINE>
           [[ 75,  76,  77,  78,  79],
            [ 80,  81,  82,  83,  84],
            [ 85,  86,  87,  88,  89],
            [ 90,  91,  92,  93,  94],
            [ 95,  96,  97,  98,  99]],
    <BLANKLINE>
           [[100, 101, 102, 103, 104],
            [105, 106, 107, 108, 109],
            [110, 111, 112, 113, 114],
            [115, 116, 117, 118, 119],
            [120, 121, 122, 123, 124]]])

    >>> sphere(X, (0, 0, 0))
    array([[[ 0,  1],
            [ 5,  6]],
    <BLANKLINE>
           [[25, 26],
            [30, 31]]])

    >>> sphere(X, (1, 1, 0))
    array([[[ 0,  1],
            [ 5,  6],
            [10, 11]],
    <BLANKLINE>
           [[25, 26],
            [30, 31],
            [35, 36]],
    <BLANKLINE>
           [[50, 51],
            [55, 56],
            [60, 61]]])

    >>> sphere(X, (4, 2, 2))
    array([[[ 81,  82,  83],
            [ 86,  87,  88],
            [ 91,  92,  93]],
    <BLANKLINE>
           [[106, 107, 108],
            [111, 112, 113],
            [116, 117, 118]]])

    >>> Y = np.arange(8).reshape(2, 4)
    >>> sphere(Y, (0, 0))
    array([[0, 1],
           [4, 5]])

    """
    
    # checking type
    assert isinstance(arr, np.ndarray), 'input array must be type np.ndarray'
    assert isinstance(c, tuple), 'center must be type tuple'
    assert isinstance(r, int), 'radius must be type int'
        
    # checking size
    assert (1 < len(c) < 4), 'tuple length must 2 or 3'
    assert len(c) == len(arr.shape), 'tuple length must equal array dim'
    assert r < min(arr.shape), 'radius must be smaller than the smallest dim\
        of the array'

    # checking valid `c` for `arr`; IndexError raised if not
    arr[c]

    b = lambda v, r: v - r
    f = lambda v, r: v + r

    # center indices
    if len(c) == 3:
        z, x, y = c[0], c[1], c[2]
        # zb, zf = z - r, z + r
        zb, zf = b(z, r), f(z, r)
    elif len(c) == 2:
        x, y = c[0], c[1] 

    # reach
    xb, xf = b(x, r), f(x, r)
    yb, yf = b(y, r), f(y, r)

    # checking within dimensions
    if len(c) == 3:
        zb, zf = inrange(arr, zb, zf, 0)
        xb, xf = inrange(arr, xb, xf, 1)
        yb, yf = inrange(arr, yb, yf, 2)
        return arr[zb:zf+1, xb:xf+1, yb:yf+1]
    elif len(c) == 2:
        xb, xf = inrange(arr, xb, xf, 0)
        yb, yf = inrange(arr, yb, yf, 1)
        return arr[xb:xf+1, yb:yf+1]



if __name__ == '__main__':
    import doctest
    doctest.testmod()
