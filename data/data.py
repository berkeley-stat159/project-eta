from __future__ import print_function, division

import hashlib
import os


def generate_file_md5(filename, blocksize=2**20):
    """Generate md5 hash for a file

    Parameters
    ----------
    filename: the name of the file for which a hash
    is to be generated

    Returns
    -------
    m.hexdigest(): concatenated strings of the md5
    hash generated for the specified file
    
    """
    m = hashlib.md5()
    with open(filename, "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()


def check_hashes(d):
    """Check whether the generated hashes match the
    corresponding hases in a dictionary

    Parameters
    ----------
    d: a dictionary with file names as keys and
    hashes as values

    Returns
    -------
    True if the generated hashes match the ones in the
    dictionary and False if not. 

    
    """
    all_good = True
    for k, v in d.items():
        digest = generate_file_md5(k)
        if v == digest:
            print("The file {0} has the correct hash.".format(k))
        else:
            print("ERROR: The file {0} has the WRONG hash!".format(k))
            all_good = False
    return all_good


def generateActualFileHashes():
    """
    Pulls actual hash values from the given text file in order to compare

    """
    hashes = {}
    with open('ds005_raw_checksums.txt', 'r') as checks:
        l = checks.readlines()
    for string in l:
        split = string.split(' ')
        hashes[split[2].rstrip('\n')] = split[0]

    return hashes    

if __name__ == "__main__":
    d = generateActualFileHashes()
    if check_hashes(d):
        print("All hashes are correct, data not corrupted.")
    else:
        print("One or more hashes are incorrect, data may be corrupted.")
