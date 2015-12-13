from __future__ import print_function, division

import hashlib
import os
import pdb
import json


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

def get_hash_for_all(dir):
    hashes = dict()
    for root, dirs, files in os.walk(dir):
        for f in files:
            file_path = os.path.join(root, f)
            hashes[file_path] = generate_file_md5(file_path)
    return hashes

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


def generateActualFileHashes(dir):
    """
    Pulls actual hash values from the given text file in order to compare

    """
    hashes = {}
    for root, dirs, files in os.walk(dir):
        for f in files:
            file_path = os.path.join(root, f)
            hashes[file_path] = generate_file_md5(file_path)
    return hashes    

if __name__ == "__main__":
    """
    hashes = get_hash_for_all('ds005')
    with open('ds005_hashes.json', 'w') as out_file:
        json.dump(hashes, out_file)
    """
    with open('ds005_hashes.json', 'r') as in_file:
        dic = json.load(in_file)
    check_hashes(dic)

