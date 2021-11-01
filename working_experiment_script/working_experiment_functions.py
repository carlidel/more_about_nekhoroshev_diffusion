import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import multiprocessing
# For parallelization
from joblib import Parallel, delayed

import nekhoroshev_tools as nt


NCORES = multiprocessing.cpu_count()


def current_estimate_backward(I_min, I_max, I_star, exponent, c, t):
    module = nt.stationary_dist(I_min, I_max, I_star, exponent, c) * 2
    ana_current = np.asarray(
        nt.current_generic(
            t, lambda x: module, I_max,
            (I_max / 3) * 2, I_star, exponent, c
        )
    )
    return ana_current


def current_estimate_forward(I_min, I_max, I_star, exponent, c, t):
    module = nt.stationary_dist(I_min, I_max, I_star, exponent, c) * 2
    def dist(x):
        if hasattr(x, "__iter__"):
            y = np.empty_like(x)
            for i, e in enumerate(x):
                y[i] = -module if e <= I_min else module * \
                    (((e - I_min) / (I_max - I_min)) - 1)
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

    low_f = scipy.interpolate.interp1d(t_low, c_low, kind="cubic")
    high_f = scipy.interpolate.interp1d(t_high, c_high, kind="cubic")

    def mid_f(x):
        return (low_f(x) + high_f(x)) / 2.0

    return low_f, mid_f, high_f


def pre_fit_sampler(data, I_step=None, samples=10, backward_flag=True, forward_flag=False, backward_upper_bound=None, backward_lower_bound=None, forward_lower_bound=None):
    if I_step is None:
        I_step = data["global"]["mov"][-1]["I_step"]
    x_list = []
    y_list = []
    for d in data["global"]["mov"]:
        if d["kind"] == "backward" and backward_flag:
            top_idx = np.argmax(d["cur"])
            if backward_lower_bound is not None:
                idx = np.argmax(
                    np.absolute(
                        (d["cur"][top_idx:] / data["interpolation"]
                         (d["t_abs"][top_idx:])) - 1
                    ) < backward_lower_bound
                )
                if idx == 0:
                    idx = len(d["cur"])
                else:
                    idx += top_idx
            else:
                idx = len(d["cur"])
            if backward_upper_bound is not None:
                beg_idx = np.argmax(
                    np.absolute(
                        (d["cur"][top_idx:] / data["interpolation"]
                         (d["t_abs"][top_idx:])) - 1
                    ) < backward_upper_bound
                )
                beg_idx += top_idx
                if beg_idx > idx:
                    beg_idx = top_idx
            else:
                beg_idx = top_idx

            f = scipy.interpolate.interp1d(
                d["t_rel"][beg_idx:idx], d["cur"][beg_idx:idx], kind="cubic"
            )
            t = np.linspace(
                d["t_rel"][beg_idx],
                d["t_rel"][idx-1],
                samples
            )
            t_global = np.linspace(
                d["t_abs"][beg_idx],
                d["t_abs"][idx-1],
                samples
            )
            cur = f(t) / data["interpolation"](t_global) - 1
            x_list.append(["backward", d["I_max"], d["I_max"] + I_step, t])
            y_list.append(cur)
        if d["kind"] == "forward" and forward_flag:
            if forward_lower_bound is not None:
                idx = np.argmax(
                    np.absolute(
                        (d["cur"] / data["interpolation"](d["t_abs"])) - 1
                    ) < forward_lower_bound
                )
                if idx == 0:
                    idx = len(d["cur"])
            else:
                idx = len(d["cur"])

            f = scipy.interpolate.interp1d(
                d["t_rel"][:idx], d["cur"][:idx], kind="cubic"
            )
            t = np.linspace(
                d["t_rel"][1],
                d["t_rel"][idx-1],
                samples
            )
            t_global = np.linspace(
                d["t_abs"][1],
                d["t_abs"][idx-1],
                samples
            )
            cur = f(t) / data["interpolation"](t_global) - 1
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
    print("Values:", "I_star", I_star, "k", params["k"].value)
    resid = np.array([])

    def compare(x, y):
        y[y==0] = 1.0
        if x[0] == "backward":
            module = nt.stationary_dist(
                x[1], x[2], I_star, exponent, c) * 2
            ana_current = np.asarray(
                nt.current_generic(
                    x[3], lambda x: module, x[2],
                    (x[2] / 3) * 2, I_star, exponent, c
                )
            )
            return ((y - ana_current) / y)
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
            return ((y - ana_current) / y)

    blocks = Parallel(NCORES)(delayed(compare)(x, y)
                          for x, y in zip(x_list, y_list))

    for b in blocks:
        resid = np.append(resid, b)

    resid[np.isinf(resid)] = 1e10
    print("Error:", np.sum(np.power(resid, 2)))
    return resid
