# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy
import scipy.integrate
import scipy.interpolate
from tqdm import tqdm
import crank_nicolson_numba.generic as cn
import itertools
import os
import multiprocessing
# For parallelization
from joblib import Parallel, delayed
import json

import lmfit
import nekhoroshev_tools as nt


# %%
def current_estimate_backward(I_min, I_max, I_star, exponent, c, t):
    module = nt.stationary_dist(I_min, I_max, I_star, exponent, c) * 2
    ana_current = np.asarray(
        nt.current_generic(
            t, lambda x: module, I_max,
            (I_max / 3) * 2, I_star, exponent, c
        )
    )
    return ana_current


# %%
def current_estimate_forward(I_min, I_max, I_star, exponent, c, t):
    module = nt.stationary_dist(
        I_min, I_max, I_star, exponent, c) * 2

    def dist(x):
        if hasattr(x, "__iter__"):
            y = np.empty_like(x)
            for i, e in enumerate(x):
                y[i] = -module if e <= I_min else module *                     (((e - I_min) / (I_max - I_min)) - 1)
            return y
        if x <= I_min:
            return - module
        else:
            return module * (((x - I_min) / (I_max - I_min)) - 1)
    ana_current = np.asarray(
        nt.current_generic(
            t, dist, I_max,
            (I_max / 3) * 2, I_star, exponent, c
        )
    )
    return ana_current

# %% [markdown]
# ## Experiment Parameters

# %%
I_star = 20.0
kappa = 0.33
exponent = 1 / (2 * kappa)

I_max_list = np.arange(0.2, 1.5, 0.1) * I_star
I_step_list = np.array([0.01, 0.005, 0.02]) * I_star
fraction_list = np.array([1.0, 0.5, 0.01, 0.001])

I_step = I_step_list[0]

I_sampling = 2500
t_sampling = 1000000

n_0_step = t_sampling
n_0_samp = 10
n_1_step = t_sampling
n_1_samp = 1
ana_samples = 500


# %%
def f(x, I_max):
    c = nt.standard_c(0.0, I_max, I_star, exponent)
    cur = np.absolute(
        current_estimate_forward(
            I_max - I_step, I_max, I_star, exponent, c, x
        )
    )
    return np.absolute(cur - 2e-3)


# %%
point = 2e-3

times = []
for I_max in tqdm(I_max_list):
    sol = scipy.optimize.minimize(f, 1.0, I_max)
    times.append(sol.x[0])


# %%
t = np.logspace(-5, 5, 21)
curs = []
for I_max in tqdm(I_max_list):
    c = nt.standard_c(0.0, I_max, I_star, exponent)
    curs.append(np.absolute(current_estimate_forward(
        I_max - I_step, I_max, I_star, exponent, c, t
    )))


# %%
with open("base_experiment.sub", 'r') as f:
    base_experiment = f.read()

base_experiment += "\n\n"
block = (
    "queue\n\n"
)

for I_max, t_max in zip(I_max_list, times):
    for I_step in I_step_list:
        for fraction in fraction_list:
            parameters = {}
            parameters["I_min"] = 3.5
            parameters["I_max"] = I_max
            parameters["movement_list"] = [
                {"kind": "still"},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
                {"kind": "backward", "mov": I_step},
                {"kind": "forward", "mov": I_step},
            ]
            parameters["I_sampling"] = I_sampling
            parameters["t_sampling"] = t_sampling
            parameters["time_interval"] = t_max / fraction
            parameters["I_star"] = I_star
            parameters["exponent"] = exponent
            parameters["c"] = nt.standard_c(0.0, I_max, I_star, exponent)
            parameters["n_0_step"] = n_0_step
            parameters["n_0_samp"] = n_0_samp
            parameters["n_1_step"] = n_1_step
            parameters["n_1_samp"] = n_1_samp
            parameters["ana_samples"] = ana_samples
            parameters["fraction_in_usage"] = fraction

            name = "Imax_{:.2f}_Istep_{:.2f}_fraction_{:.3f}".format(
                I_max, I_step, fraction)
            #print(name)
            parameters["name"] = name

            with open("parameters/param_{}.json".format(name), 'w') as f:
                json.dump(parameters, f, indent=4)
            
            base_experiment += "file=" + "param_{}.json".format(name) + "\n" + block

with open("execute_experiment.sub", 'w') as f:
    f.write(base_experiment)


