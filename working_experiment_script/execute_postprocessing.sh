#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
export MYPYTHON=/afs/cern.ch/work/c/camontan/public/miniconda3

unset PYTHONHOME
unset PYTHONPATH
source $MYPYTHON/bin/activate
export PATH=$MYPYTHON/bin:$PATH

which python

python3 postprocessing.py $1