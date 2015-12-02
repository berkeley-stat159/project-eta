#!/bin/bash

DIR='./ds005'

if test -d $DIR; then
    echo "directory '$DIR' already exists"
else
    wget http://openfmri.s3.amazonaws.com/tarballs/ds005_raw.tgz
    wget http://openfmri.s3.amazonaws.com/tarballs/ds005_raw_checksums.txt
    tar xvzf ds005_raw.tgz 
fi
