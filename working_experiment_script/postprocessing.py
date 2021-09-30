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

NCORES = multiprocessing.cpu_count()
FRACTION_LIST_BEFORE = np.array([0.0, 0.001, 0.01])
FRACTION_LIST_AFTER = np.array([0.1, 0.05, 0.01, 0.005, 0.001])
PATH = "/eos/project-d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"

def interpolation_system(data_list):
    t_low = np.array([])
    c_low = np.array([])
    t_high = np.array([])
    c_high = np.array([])
    for i, d in enumerate(data_list):
        if i == len(data_list) - 1:
            t_low = np.append(t_low, d["t_abs"][-1])
            t_high = np.append(t_high, d["t_abs"][-1])
            c_low = np.append(c_low, d["cur"][-1])
            c_high = np.append(c_high, d["cur"][-1])
        elif d["kind"] == "still":
            t_low = np.append(t_low, d["t_abs"])
            t_high = np.append(t_high, d["t_abs"])
            c_low = np.append(c_low, d["cur"])
            c_high = np.append(c_high, d["cur"])
        elif d["kind"] == "backward":
            t_high = np.append(t_high, d["t_abs"][-1])
            c_high = np.append(c_high, d["cur"][-1])
        elif d["kind"] == "forward":
            t_low = np.append(t_low, d["t_abs"][-1])
            c_low = np.append(c_low, d["cur"][-1])

    low_f = scipy.interpolate.interp1d(t_low, c_low, kind="linear")
    high_f = scipy.interpolate.interp1d(t_high, c_high, kind="linear")

    def mid_f(x):
        return (low_f(x) + high_f(x)) / 2.0

    return low_f, mid_f, high_f


def pre_fit_sampler(data, I_step, samples=10, fraction_before=0.0, fraction_after=1.0, backward_flag=True, forward_flag=False):
    assert fraction_before < fraction_after
    x_list = []
    y_list = []
    for d in data["global"]["mov"]:
        f = scipy.interpolate.interp1d(
            d["t_rel"], d["cur"], kind="cubic"
        )

        beg = d["t_rel"][1] + (d["t_rel"][-1] - d["t_rel"][1]) * fraction_before
        end = d["t_rel"][1] + (d["t_rel"][-1] - d["t_rel"][1]) * fraction_after
        t = np.linspace(beg, end, samples)

        beg = d["t_abs"][1] + (d["t_abs"][-1] - d["t_abs"][1]) * fraction_before
        end = d["t_abs"][1] + (d["t_abs"][-1] - d["t_abs"][1]) * fraction_after
        t_global = np.linspace(beg, end, samples)

        cur = f(t) / data["interpolation"](t_global) - 1

        if d["kind"] == "backward" and backward_flag:
            x_list.append(["backward", d["I_max"], d["I_max"] + I_step, t])
            y_list.append(cur)
        if d["kind"] == "forward" and forward_flag:
            x_list.append(["forward", d["I_max"] - I_step, d["I_max"], t])
            y_list.append(cur)
    return x_list, y_list


def forward_dist(x, module, I_max_old, I_max):
    if hasattr(x, "__iter__"):
        y = np.empty_like(x)
        for i, e in enumerate(x):
            y[i] = -module if e <= I_max_old else module * \
                (((e - I_max_old) / (I_max - I_max_old)) - 1)
        return y
    if x <= I_max_old:
        return - module
    else:
        return module * (((x - I_max_old) / (I_max - I_max_old)) - 1)


def resid_func(params, x_list, y_list):
    I_star = params["I_star"].value
    exponent = 1 / (params["k"].value * 2)
    c = params["c"].value
    print(I_star, params["k"].value)
    resid = np.array([])

    def compare(x, y):
        if x[0] == "backward":
            module = nt.stationary_dist(
                x[1], x[2], I_star, exponent, c) * 2
            ana_current = np.asarray(
                nt.current_generic(
                    x[3], lambda x: module, x[2],
                    (x[2] / 3) * 2, I_star, exponent, c
                )
            )
            y_den = y.copy()
            y_den[y_den == 0.0] = 1.0
            return (y - ana_current) / y_den
        elif x[0] == "forward":
            module = nt.stationary_dist(
                x[1], x[2], I_star, exponent, c) * 2
            ana_current = np.asarray(
                nt.current_generic(
                    x[3],
                    lambda p: forward_dist(p, module, x[1], x[2]),
                    x[2],
                    (x[2] / 3) * 2,
                    I_star, exponent, c
                )
            )
            y_den = y.copy()
            y_den[y_den == 0.0] = 1.0
            return (y - ana_current) / y_den

    blocks = Parallel(NCORES)(delayed(compare)(x, y)
                              for x, y in zip(x_list, y_list))

    for b in blocks:
        resid = np.append(resid, b)

    resid[np.isinf(resid)] = 1e10
    print(np.sum(np.power(resid, 2)))
    return resid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input_data',
        metavar='path',
        type=str,
        help='pkl filename with values to use'
    )
    args = parser.parse_args()

    FILE = args.input_data
    
    with open(os.path.join(PATH, FILE), 'rb') as f:
        parameters, data = pickle.load(f)

    data['interpolation'] = interpolation_system(data["global"]["mov"])[1]

    pd = {}
    pd["info"] = parameters


    frac_list = list(filter(lambda x: x[0] < x[1], itertools.product(
        FRACTION_LIST_BEFORE, FRACTION_LIST_AFTER
    )))
    fit_methods = [
        ("all", True, True),
        ("forward_only", True, False),
#        ("backward_only", False, True),
    ]

    for fraction_before, fraction_after in tqdm(frac_list):
        print((fraction_before, fraction_after))
        pd[(fraction_before, fraction_after)] = {}
        for method, forward_flag, backward_flag in fit_methods:
            print(method)
            x_list, y_list = pre_fit_sampler(
                data,
                parameters["movement_list"][-1]["mov"],
                fraction_before=fraction_before,
                fraction_after=fraction_after,
                samples=10,
                forward_flag=forward_flag,
                backward_flag=backward_flag
            )
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

            pd[(fraction_before, fraction_after)][method] = result
    
    container = {
        parameters["I_max"]: {
            parameters["movement_list"][-1]["mov"]: {
                parameters["fraction_in_usage"]: pd
            }
        }
    }

    with open("FIT_" + os.path.basename(FILE), 'wb') as f:
        pickle.dump(container, f)

    subprocess.run([
        "eos",
        "cp",
        "*pkl",
        "/eos/project/d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"
    ])
