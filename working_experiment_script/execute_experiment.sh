#!/bin/bash

export EOS_MGM_URL=root://eosuser.cern.ch
export MYPYTHON=/afs/cern.ch/work/c/camontan/public/anaconda3

unset PYTHONHOME
unset PYTHONPATH
source $MYPYTHON/bin/activate
export PATH=$MYPYTHON/bin:$PATH

which python

python3 experiment.py $1 --no-stationary

eos cp *.pkl /eos/project/d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data