import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import crank_nicolson_numba.generic as cn
import itertools
import os
import multiprocessing
import argparse
import datetime
import json
import pickle
from tqdm import tqdm
# For parallelization

import nekhoroshev_tools as nt

def experiment_routine(
    I_min,
    I_max,
    movement_list,
    I_sampling,
    t_sampling,
    time_interval,
    I_star,
    exponent,
    c,
    n_0_step,
    n_0_samp,
    n_1_step,
    n_1_samp,
    ana_samples,
    stationary_flag=True
):
    I_max_0 = I_max
    I_max_old = I_max
    data = {}

    data["stationary"] = {}
    data["stationary"]["imm"] = []
    data["stationary"]["mov"] = []
    data["global"] = {}
    data["global"]["imm"] = []
    data["global"]["mov"] = []
    data["analytic"] = []

    I, dI = np.linspace(I_min, I_max, I_sampling,
                        endpoint=False, retstep=True)
    rho_0 = np.array([
        nt.stationary_dist(i, I_max, I_star, exponent, c) for i in I
    ])
    engine_s_imm = cn.cn_generic(
        I_min, I_max,
        rho_0,
        time_interval / t_sampling,
        lambda x: nt.D(x, I_star, exponent, c, True)
    )
    engine_s_imm.lock_left()
    engine_s_mov = cn.cn_generic(
        I_min, I_max,
        rho_0,
        time_interval / t_sampling,
        lambda x: nt.D(x, I_star, exponent, c, True)
    )
    engine_s_mov.lock_left()

    I, dI = np.linspace(0.0, I_max, I_sampling,
                        endpoint=False, retstep=True)
    rho_0 = np.exp(-I)
    # will this improve things?
    idx = np.argmin(I / I_star < 0.4)
    if idx != 0:
        s_dist = nt.stationary_dist(I[idx:], I_max, I_star, exponent, c)
        s_dist = s_dist / s_dist[0] * rho_0[idx]
        rho_0[idx:] = s_dist

    engine_g_imm = cn.cn_generic(
        0.0, I_max,
        rho_0,
        time_interval / t_sampling,
        lambda x: nt.D(x, I_star, exponent, c, True)
    )
    engine_g_mov = cn.cn_generic(
        0.0, I_max,
        rho_0,
        time_interval / t_sampling,
        lambda x: nt.D(x, I_star, exponent, c, True)
    )

    def current(step, samp, kind, I_step=np.nan):
        if stationary_flag:
            rho_s_imm = engine_s_imm.get_data_with_x()
            t_s_imm, c_s_imm = engine_s_imm.analytical_current(
                step, samp)
            data["stationary"]["imm"].append({
                "rho_before": rho_s_imm,
                "t_abs": t_s_imm,
                "cur": c_s_imm,
                "kind": kind,
                "I_step": I_step
            })
            rho_s_mov = engine_s_mov.get_data_with_x()
            t_s_mov, c_s_mov = engine_s_mov.analytical_current(
                step, samp)
            data["stationary"]["mov"].append({
                "rho_before": rho_s_mov,
                "t_abs": t_s_mov,
                "cur": c_s_mov,
                "kind": kind,
                "I_step": I_step
            })
        rho_g_imm = engine_g_imm.get_data_with_x()
        t_g_imm, c_g_imm = engine_g_imm.analytical_current(
            step, samp)
        data["global"]["imm"].append({
            "rho_before": rho_g_imm,
            "t_abs": t_g_imm,
            "cur": c_g_imm,
            "kind": kind,
            "I_step": I_step
        })
        rho_g_mov = engine_g_mov.get_data_with_x()
        t_g_mov, c_g_mov = engine_g_mov.analytical_current(
            step, samp)
        data["global"]["mov"].append({
            "rho_before": rho_g_mov,
            "t_abs": t_g_mov,
            "cur": c_g_mov,
            "kind": kind,
            "I_step": I_step
        })

    def catalog():
        if stationary_flag:
            data["stationary"]["imm"][-1]["I_max"] = engine_s_imm.I_max
            data["stationary"]["imm"][-1]["rho"] = engine_s_imm.get_data_with_x()
            data["stationary"]["imm"][-1]["t_rel"] = (
                data["stationary"]["imm"][-1]["t_abs"]
                - data["stationary"]["imm"][-1]["t_abs"][0]
            )

            data["stationary"]["mov"][-1]["I_max"] = engine_s_mov.I_max
            data["stationary"]["mov"][-1]["rho"] = engine_s_mov.get_data_with_x()
            data["stationary"]["mov"][-1]["t_rel"] = (
                data["stationary"]["mov"][-1]["t_abs"]
                - data["stationary"]["mov"][-1]["t_abs"][0]
            )

        data["global"]["imm"][-1]["I_max"] = engine_g_imm.I_max
        data["global"]["imm"][-1]["rho"] = engine_g_imm.get_data_with_x()
        data["global"]["imm"][-1]["t_rel"] = (
            data["global"]["imm"][-1]["t_abs"]
            - data["global"]["imm"][-1]["t_abs"][0]
        )

        data["global"]["mov"][-1]["I_max"] = engine_g_mov.I_max
        data["global"]["mov"][-1]["rho"] = engine_g_mov.get_data_with_x()
        data["global"]["mov"][-1]["t_rel"] = (
            data["global"]["mov"][-1]["t_abs"]
            - data["global"]["mov"][-1]["t_abs"][0]
        )

    for mov in tqdm(movement_list):
        if mov["kind"] == "still":
            current(n_0_step, n_0_samp, "still")
            catalog()
            data["analytic"].append({"kind": "none"})

        elif mov["kind"] == "forward":
            I_max = I_max_old + mov["mov"]
            if stationary_flag:
                engine_s_mov.move_barrier_forward(mov["mov"], resample=True)
            engine_g_mov.move_barrier_forward(mov["mov"], resample=True)
            current(n_1_step, n_1_samp, "forward", mov["mov"])
            catalog()

            t_samples = np.linspace(
                data["global"]["imm"][-1]["t_rel"][1],
                data["global"]["imm"][-1]["t_rel"][-1],
                ana_samples
            )
            t_samples_abs = np.linspace(
                data["global"]["imm"][-1]["t_abs"][1],
                data["global"]["imm"][-1]["t_abs"][-1],
                ana_samples
            )
            module = nt.stationary_dist(
                I_max_old, I_max, I_star, exponent, c) * 2

            def dist(x):
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
            ana_current = np.asarray(
                nt.current_generic(
                    t_samples, dist, I_max,
                    (I_max / 3) * 2, I_star, exponent, c
                )
            )
            data["analytic"].append({
                "kind": "backward",
                "t_rel": t_samples,
                "t_abs": t_samples_abs,
                "cur": ana_current
            })
            I_max_old = I_max

        elif mov["kind"] == "backward":
            I_max = I_max_old - mov["mov"]
            if stationary_flag:
                engine_s_mov.move_barrier_backward(mov["mov"], resample=True)
            engine_g_mov.move_barrier_backward(mov["mov"], resample=True)
            current(n_1_step, n_1_samp, "backward", mov["mov"])
            catalog()

            t_samples = np.linspace(
                data["global"]["imm"][-1]["t_rel"][1],
                data["global"]["imm"][-1]["t_rel"][-1],
                ana_samples
            )
            t_samples_abs = np.linspace(
                data["global"]["imm"][-1]["t_abs"][1],
                data["global"]["imm"][-1]["t_abs"][-1],
                ana_samples
            )
            module = nt.stationary_dist(
                I_max, I_max_old, I_star, exponent, c) * 2
            ana_current = np.asarray(
                nt.current_generic(
                    t_samples, lambda x: module, I_max,
                    (I_max / 3) * 2, I_star, exponent, c
                )
            )
            data["analytic"].append({
                "kind": "backward",
                "t_rel": t_samples,
                "t_abs": t_samples_abs,
                "cur": ana_current
            })
            I_max_old = I_max

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_json',
        metavar='path',
        type=str,
        help='JSON filepath with values to use'
    )
    parser.add_argument('--stationary', dest='stationary', action='store_true')
    parser.add_argument('--no-stationary', dest='stationary', action='store_false')
    parser.set_defaults(stationary=True)

    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        parameters = json.load(f)

    start = datetime.datetime.now()
    print("Start at:", start)

    data = experiment_routine(
        parameters["I_min"],
        parameters["I_max"],
        parameters["movement_list"],
        parameters["I_sampling"],
        parameters["t_sampling"],
        parameters["time_interval"],
        parameters["I_star"],
        parameters["exponent"],
        parameters["c"],
        parameters["n_0_step"],
        parameters["n_0_samp"],
        parameters["n_1_step"],
        parameters["n_1_samp"],
        parameters["ana_samples"],
        args.stationary
    )

    end = datetime.datetime.now()
    print("End at:", end)
    print("Elapsed time:", end-start)

    with open("working_experiment_" + parameters["name"] + ".pkl", 'wb') as f:
        pickle.dump((parameters, data), f)