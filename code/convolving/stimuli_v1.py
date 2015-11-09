""" Functions to work with standard OpenFMRI stimulus files

The functions have docstrings according to the numpy docstring standard - see:

    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt
"""

import numpy as np

def events2neural(taskfname, tr, n_trs):
    """ Return predicted neural time course from event file `task_fname`

    Parameters
    ----------
    task_fname : str
        Filename of event file
    tr : float
        TR in seconds
    n_trs : int
        Number of TRs in functional run

    Returns
    -------
    time_course : array shape (n_trs,)
        Predicted neural time course, one value per TR
    """
    task = [line.strip().split('\t') for line in open(taskfname)][1:]
    # Check that the file is plausibly a task file
    #if task.ndim != 2 or task.shape[1] != 3:
    #    raise ValueError("Is {0} really a task file?", task_fname)
    # Convert onset, duration seconds to TRs
    for i in range(len(task)):
        if tr[i] == 0:
            tr[i] = 1
        task[i][0] = int(float(task[i][0])/tr[i])
        task[i][1] = int(float(task[i][1])/tr[i])
            
    # Neural time course from onset, duration, amplitude for each event
    time_course = np.zeros(n_trs)
    for line in task:
        time_course[line[0]:(line[0] + line[1])] = line[2]
    return time_course
