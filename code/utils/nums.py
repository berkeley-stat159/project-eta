def n_convert(*args):
    """Convert an arbitrary number of values using `n_to_string`

    Parameters
    ----------
    *args : variable length argument list

    Returns
    -------
    n_converted : tuple
        tuple of length *args with converted values

    Examples
    --------
    >>> n_convert(1)
    '001'
    >>> n_convert(10)
    '010'
    >>> n_convert(5, 6, 7)
    ('005', '006', '007')
    >>> n_convert(1, 42, 999)
    ('001', '042', '999')
    """
    n_converted = tuple([n_to_string(n) for n in args])
    if len(n_converted) == 1:
        return n_converted[0]
    else:
        return n_converted

def n_to_string(n):
    """Converting int to three-character string

    Parameters
    ----------
    n : int
        The number to be converted

    Returns
    -------
    A three-character string

    Examples
    --------
    >>> n_to_string(0)
    '000'
    >>> n_to_string(1)
    '001'
    >>> n_to_string(10)
    '010'
    >>> n_to_string(99)
    '099'
    """
    assert type(n) is int, '`n` must be of type int'
    assert n < 1000, 'maximum number of digits is three'
    return str(n).zfill(3)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
