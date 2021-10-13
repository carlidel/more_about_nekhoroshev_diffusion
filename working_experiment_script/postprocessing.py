import pickle
import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
from tqdm import tqdm
import crank_nicolson_numba.generic as cn
import itertools
import os
import multiprocessing
import subprocess
import json
import argparse
# For parallelization
from joblib import Parallel, delayed

import lmfit
import nekhoroshev_tools as nt

from working_experiment_functions import *

UPPER_BOUNDS = np.array([1.0])
LOWER_BOUNDS = np.array([0.1, 0.01, 0.001])

PATH = "/eos/project-d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_data',
        metavar='path',
        type=str,
        help='pkl filename with values to use'
    )
    parser.add_argument(
        'upper_bound',
        type=float
    )
    parser.add_argument(
        'lower_bound',
        type=float
    )
    parser.add_argument('--backward', dest='backward', action='store_true')
    parser.add_argument('--no-backward', dest='backward', action='store_false')
    parser.set_defaults(backward=False)
    parser.add_argument('--forward', dest='forward', action='store_true')
    parser.add_argument('--no-forward', dest='forward', action='store_false')
    parser.set_defaults(forward=True)

    parser.add_argument('--partial', type=int, default=-1)

    args = parser.parse_args()

    FILE = args.input_data
    up_bound = args.upper_bound
    low_bound = args.lower_bound
    assert(up_bound > low_bound)
    forward_flag = args.forward
    backward_flag = args.backward
    partial = args.partial

    assert(forward_flag or backward_flag)
    
    if forward_flag and backward_flag:
        method = "all"
    elif forward_flag and not backward_flag:
        method = "forward_only"
    else:
        method = "backward_only"

    with open(os.path.join(PATH, FILE), 'rb') as f:
         parameters, data = pickle.load(f)

    data['interpolation'] = interpolation_system(data["global"]["mov"])[1]

    pd = {}
    pd["info"] = parameters
    pd[(up_bound, low_bound)] = {}

    print((up_bound, low_bound))
    print(method)

    x_list, y_list = pre_fit_sampler(
        data,
        samples=20,
        forward_flag=forward_flag,
        backward_flag=backward_flag,
        backward_upper_bound=up_bound, 
        backward_lower_bound=low_bound,
        forward_lower_bound=low_bound
    )

    if partial == -1:
        partial = len(x_list)
        print("Set usage of all data!")
    elif partial > len(x_list):
        partial = len(x_list)
        print("Set usage of all data!")
    else:
        print("Using {} samples out of {}".format(partial, len(x_list)))
    
    orig_len = len(x_list)
    x_list = x_list[:partial]
    y_list = y_list[:partial]

    fit_parameters = lmfit.Parameters()
    fit_parameters.add(
        "I_star", value=parameters["I_star"], vary=True, min=0.1)
    fit_parameters.add(
        "k", value=1 / parameters["exponent"] / 2, vary=True, min=0.0)
    fit_parameters.add(
        "c", value=parameters["c"], vary=False)
    
    try:
        result = lmfit.minimize(
            resid_func,
            fit_parameters,
            args=(x_list, y_list)
        )
    except Exception as e:
        print("FAILED!")
        result = "FAILED!"

    pd[(up_bound, low_bound)][method] = result

    if partial == orig_len:
        container = {
            parameters["I_max"]: {
                parameters["movement_list"][-1]["mov"]: {
                    parameters["fraction_in_usage"]: pd
                }
            }
        }
    else:
        container = {
            parameters["I_max"]: {
                parameters["movement_list"][-1]["mov"]: {
                    parameters["fraction_in_usage"]: {
                        partial: pd
                    }
                }
            }
        }

    OUTFILE = (
        "FIT_ub_{}_lb_{}_md_{}_".format(up_bound, low_bound, method) 
        + ("par_{}_".format(partial) if partial != orig_len else "")
        + os.path.basename(FILE).replace("working_experiment_", "")
    )

    with open(OUTFILE, 'wb') as f:
        pickle.dump(container, f)
