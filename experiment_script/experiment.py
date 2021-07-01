import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
import scipy.interpolate
from tqdm import tqdm
import crank_nicolson_numba.generic as cn
import itertools
from joblib import Parallel, delayed
import datetime
import pickle
import argparse
import json

import nekhoroshev_tools as nt

################################################################################
######   PROTOCOLS   ###########################################################
################################################################################

def p_relax(samples, it_per_sample):
    return [
        {"movement": "still", "samples": samples,
            "it_per_sample": it_per_sample}
    ]


def p1(step_size, samples, relax_samples, it_per_sample, steps_in_I, repetitions):
    temp = []

    for i in range(repetitions):
        for j in range(steps_in_I):
            temp.append(
                {"movement": "forward", "amount": step_size, "samples": samples,
                    "it_per_sample": it_per_sample, "label": i, "sub_label": j}
            )
        for j in range(steps_in_I):
            temp.append(
                {"movement": "backward", "amount": step_size, "samples": samples,
                    "it_per_sample": it_per_sample, "label": i, "sub_label": j}
            )

    return p_relax(relax_samples, it_per_sample) + temp


def p2(step_size, samples, relax_samples, it_per_sample, steps_in_I, repetitions):
    temp = []

    for i in range(steps_in_I):
        for j in range(repetitions):
            temp.append(
                {"movement": "forward", "amount": step_size, "samples": samples,
                    "it_per_sample": it_per_sample, "label": i, "sub_label": j}
            )
            temp.append(
                {"movement": "backward", "amount": step_size, "samples": samples,
                    "it_per_sample": it_per_sample, "label": i, "sub_label": j}
            )
        temp.append(
            {"movement": "forward", "amount": step_size, "samples": samples,
                "it_per_sample": it_per_sample, "label": i, "sub_label": "final"}
        )

    return p_relax(relax_samples, it_per_sample) + temp


def p3(step_size, samples, relax_samples, it_per_sample, steps_in_I, repetitions):
    temp = []

    for i in range(steps_in_I):
        for j in range(repetitions):
            temp.append(
                {"movement": "forward", "amount": step_size * (i + 1), 
                    "samples": samples, "it_per_sample": it_per_sample, 
                    "label": i, "sub_label": j}
            )
            temp.append(
                {"movement": "backward", "amount": step_size * (i + 1), 
                    "samples": samples, "it_per_sample": it_per_sample, 
                    "label": i, "sub_label": j}
            )
        temp.append(
            {"movement": "backward", "amount": step_size / 2,
             "samples": samples, "it_per_sample": it_per_sample,
             "label": i, "sub_label": "final"}
        )
    
    return p_relax(relax_samples, it_per_sample) + temp

p_list = [p1, p2, p3]
p_name_list = [
    "the_long_ladder",
    "the_steady_tip-tap",
    "the_increasing_jumps"
]

################################################################################
######   FUNCTIONS   ###########################################################
################################################################################

def rho_0(I, damping_position=np.nan, l=np.nan):
    if np.isnan(damping_position) or np.isnan(l):
        return np.exp(-I)
    else:
        return np.exp(-I) / (1 + np.exp((I - damping_position)/l))


def perform_experiment(parameters, movement_list, immovable=False):
    I_list, dI = np.linspace(
        0.0, parameters["I_max"], parameters["cn_sampling"], retstep=True)
    engine = cn.cn_generic(
        0.0,
        parameters["I_max"],
        rho_0(I_list, parameters["I_damping"], dI*5),
        parameters["dt"],
        lambda x: nt.D(
            x,
            parameters["I_star"],
            parameters["exponent"],
            parameters["c"],
            True
        )
    )
    data_list = []
    absolute_time = 0.0
    for move in tqdm(movement_list):
        data = {}

        if not immovable:
            if move["movement"] == "forward":
                data["I_max_before"] = engine.I_max
                data["I_max_low"] = engine.I_max
                engine.move_barrier_forward(move["amount"])
                data["I_max_after"] = engine.I_max
                data["I_max_high"] = engine.I_max
            elif move["movement"] == "backward":
                data["I_max_before"] = engine.I_max
                data["I_max_high"] = engine.I_max
                engine.move_barrier_backward(move["amount"])
                data["I_max_after"] = engine.I_max
                data["I_max_low"] = engine.I_max
            else:
                data["I_max_before"] = engine.I_max
                data["I_max_high"] = engine.I_max
                data["I_max_after"] = engine.I_max
                data["I_max_low"] = engine.I_max
        else:
            data["I_max_before"] = engine.I_max
            data["I_max_high"] = engine.I_max
            data["I_max_after"] = engine.I_max
            data["I_max_low"] = engine.I_max

        time, current = engine.current(move["samples"], move["it_per_sample"])

        data["t_absolute"] = time
        data["t_relative"] = time - absolute_time
        absolute_time = time[-1]

        data["current"] = current

        data_list.append(data)
    return data_list


################################################################################
######   MAIN   ################################################################
################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_json',
        metavar='path',
        type=str,
        help='JSON filepath with values to use'
    )

    parser.add_argument(
        'protocol_index',
        type=int,
        help='JSON filepath with values to use'
    )

    parser.add_argument('-i',
                        '--immovable',
                        action='store_true',
                        help='Perform immovable variant of the experiment')

    parser.add_argument('-m',
                        '--movable',
                        action='store_true',
                        help='DUMMY USELESS FLAG!!!')
    
    args = parser.parse_args()

    with open(args.input_json, 'r') as f:
        parameters = json.load(f)

    parameters["exponent"] = 1/(parameters["k"]*2)

    parameters["c"] = nt.standard_c(
        0.0, parameters["I_max"], parameters["I_star"], parameters["exponent"])

    parameters["dt"] = nt.afpt(
        parameters["I_max"],
        parameters["I_max"] + parameters["step_size"],
        parameters["I_star"],
        parameters["exponent"],
        parameters["c"]
    ) / parameters["dt_sampling"]

    movement_list = p_list[args.protocol_index](
        parameters["step_size"],
        parameters["samples"],
        parameters["relax_samples"],
        parameters["it_per_sample"],
        parameters["steps_in_I"],
        parameters["repetitions"]
    )
    parameters["protocol_name"] = p_name_list[args.protocol_index]
    
    start = datetime.datetime.now()
    print("Start at:", start)

    data = perform_experiment(
        parameters,
        movement_list,
        immovable=args.immovable
    )

    end = datetime.datetime.now()
    print("End at:", end)
    print("Elapsed time:", end-start)

    with open(
        "data_" + 
        parameters["name"] + "_" + 
        parameters["protocol_name"] +
        ("_immovable" if args.immovable else "_standard") +
        ".pkl", 'wb') as f:
        pickle.dump(
            (parameters, movement_list, data),
            f
        )
