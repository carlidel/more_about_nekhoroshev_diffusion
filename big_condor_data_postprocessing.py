import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.optimize
from tqdm import tqdm
import crank_nicolson_numba.generic as cn
import itertools
# For parallelization
from joblib import Parallel, delayed
from datetime import datetime
import pickle
import matplotlib
import os
import lmfit

import nekhoroshev_tools as nt
import poly_tools as pt
import expo_tools as et

DATA_PATH = "/eos/project/d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"

MAGIC_NUMBER_HIGH = 1.01
MAGIC_NUMBER_LOW = 0.99
DISTANCE_FROM_TRUE_MAGIC = 0.002

def interpolation_system(data):
    low_points_time = []
    low_points_value = []
    high_points_time = []
    high_points_value = []
    for i, d in enumerate(data[2]):
        if d["I_max_before"] == d["I_max_after"]:
            low_points_time.append(d["t_absolute"][-1])
            high_points_time.append(d["t_absolute"][-1])

            low_points_value.append(d["current"][-1])
            high_points_value.append(d["current"][-1])

        elif d["I_max_before"] < d["I_max_after"]:
            low_points_time.append(d["t_absolute"][-1])
            low_points_value.append(d["current"][-1])

        elif d["I_max_before"] > d["I_max_after"]:
            high_points_time.append(d["t_absolute"][-1])
            high_points_value.append(d["current"][-1])

        if i == len(data[2]) - 1:
            if d["I_max_before"] < d["I_max_after"]:
                high_points_time.append(d["t_absolute"][-1])
                high_points_value.append(d["current"][-1])
            else:
                low_points_time.append(d["t_absolute"][-1])
                low_points_value.append(d["current"][-1])

    low_points_f = scipy.interpolate.interp1d(
        low_points_time,
        low_points_value,
        kind='cubic'
    )

    high_points_f = scipy.interpolate.interp1d(
        high_points_time,
        high_points_value,
        kind='cubic'
    )

    def mid_points(t):
        return (high_points_f(t) + low_points_f(t)) / 2

    return(low_points_f, mid_points, high_points_f)


def add_normed_current(data):
    low_points_f, mid_points, high_points_f = interpolation_system(data)
    for i, d in enumerate(data[2]):
        if d["I_max_before"] != d["I_max_after"]:
            d["n_current"] = d["current"]/mid_points(d["t_absolute"])
            d["n_current_low"] = d["current"]/low_points_f(d["t_absolute"])
            d["n_current_high"] = d["current"]/high_points_f(d["t_absolute"])
    return data


def extract_values_high(data, magic_number_high):
    start_points = set()
    for d in data[2]:
        if d["I_max_before"] > d["I_max_after"]:
            start_points.add(d["I_max_before"])
    start_points = list(sorted(list(start_points)))
    high_data = {
        i: {"mid": [], "low": [], "high": []}
        for i in start_points
    }
    for d in data[2]:
        if "n_current" in d and d["I_max_before"] > d["I_max_after"]:
            temp = np.argmin(np.absolute(magic_number_high - d["n_current"]))
            high_data[d["I_max_before"]]["mid"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number_high - d["n_current_low"]))
            high_data[d["I_max_before"]]["low"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number_high - d["n_current_high"]))
            high_data[d["I_max_before"]]["high"].append(
                d["t_relative"][temp]
            )
    return high_data


def extract_values_low(data, magic_number_low):
    start_points = set()
    for d in data[2]:
        if d["I_max_before"] < d["I_max_after"]:
            start_points.add(d["I_max_before"])
    start_points = list(sorted(list(start_points)))
    low_data = {
        i: {"mid": [], "low": [], "high": []}
        for i in start_points
    }
    for d in data[2]:
        if "n_current" in d and d["I_max_before"] < d["I_max_after"]:
            temp = np.argmin(np.absolute(magic_number_low - d["n_current"]))
            low_data[d["I_max_before"]]["mid"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number_low - d["n_current_low"]))
            low_data[d["I_max_before"]]["low"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number_low - d["n_current_high"]))
            low_data[d["I_max_before"]]["high"].append(
                d["t_relative"][temp]
            )
    return low_data


def make_meta_high(data, point_data, magic_value):
    val = []
    avg = []
    std = []
    avg_low = []
    std_low = []
    avg_high = []
    std_high = []
    afpt = []
    for key in point_data:
        val.append(key)

        avg.append(np.average(point_data[key]["mid"]))
        std.append(np.std(point_data[key]["mid"]))

        avg_low.append(np.average(point_data[key]["low"]))
        std_low.append(np.std(point_data[key]["low"]))

        avg_high.append(np.average(point_data[key]["high"]))
        std_high.append(np.std(point_data[key]["high"]))

        afpt.append(
            nt.afpt(
                key - data[0]["step_size"],
                key,
                data[0]["I_star"],
                data[0]["exponent"],
                data[0]["c"]
            )
        )
    return {
        "magic": magic_value,
        "val": np.asarray(val),
        "avg": np.asarray(avg),
        "std": np.asarray(std),
        "avg_low": np.asarray(avg_low),
        "std_low": np.asarray(std_low),
        "avg_high": np.asarray(avg_high),
        "std_high": np.asarray(std_high),
        "afpt": np.asarray(afpt),
    }


def make_meta_low(data, point_data, magic_value):
    val = []
    avg = []
    std = []
    avg_low = []
    std_low = []
    avg_high = []
    std_high = []
    afpt = []
    for key in point_data:
        val.append(key)

        avg.append(np.average(point_data[key]["mid"]))
        std.append(np.std(point_data[key]["mid"]))

        avg_low.append(np.average(point_data[key]["low"]))
        std_low.append(np.std(point_data[key]["low"]))

        avg_high.append(np.average(point_data[key]["high"]))
        std_high.append(np.std(point_data[key]["high"]))

        afpt.append(
            nt.afpt(
                key - data[0]["step_size"],
                key,
                data[0]["I_star"],
                data[0]["exponent"],
                data[0]["c"]
            )
        )
    return {
        "magic": magic_value,
        "val": np.asarray(val),
        "avg": np.asarray(avg),
        "std": np.asarray(std),
        "avg_low": np.asarray(avg_low),
        "std_low": np.asarray(std_low),
        "avg_high": np.asarray(avg_high),
        "std_high": np.asarray(std_high),
        "afpt": np.asarray(afpt),
    }


def test_magic_number_high(val, data):
    high_data = extract_values_high(data, val)
    meta_high_data = make_meta_high(data, high_data, val)
    result = np.sum(np.absolute(np.log10(
        meta_high_data["avg"]) - np.log10(np.absolute(meta_high_data["afpt"]))))
    return(result)


def test_magic_number_low(val, data):
    low_data = extract_values_low(data, val)
    meta_low_data = make_meta_low(data, low_data, val)
    result = np.sum(np.absolute(np.log10(
        meta_low_data["avg"]) - np.log10(np.absolute(meta_low_data["afpt"]))))
    return(result)


def find_perfect_magic_number_high(data):
    result = scipy.optimize.minimize(
        test_magic_number_high,
        1.05,
        args=(data,),
        bounds=((1.000001, 1.2),),
        method="Powell"
    )
    return result.x

def find_perfect_magic_number_low(data):
    result = scipy.optimize.minimize(
        test_magic_number_low,
        0.99,
        args=(data,),
        bounds=((0.9, 0.99999),),
        method="Powell"
    )
    return result.x


def decompose_filename(filename):
    # data_low_I_a_2_r_50_the_steady_tip-tap_standard.pkl
    splits = filename.split("_")
    I_a_position = splits[1]
    version = splits[4]
    repetitions = splits[6]
    protocol = "_".join(splits[7:])[:-4]
    return (version, protocol, I_a_position, repetitions)

if __name__ == "__main__":
    processed_data = {}
    
    files = list(sorted(os.listdir(DATA_PATH)))
    
    for f in tqdm(list(filter(lambda f: "I_a_3" in f and "immovable" not in f, files))):
        print("Processing", f)
        version, protocol, I_a_position, repetitions = decompose_filename(f)

        with open(os.path.join(DATA_PATH, f), 'rb') as f:
            data = pickle.load(f)
        
        data = add_normed_current(data)

        if protocol not in processed_data:
            processed_data[protocol] = {}

        if I_a_position not in processed_data[protocol]:
            processed_data[protocol][I_a_position] = {}

        if repetitions not in processed_data[protocol][I_a_position]:
            processed_data[protocol][I_a_position][repetitions] = {}

        true_magic_high = find_perfect_magic_number_high(data)

        magic_high = {
            "high_standard" : MAGIC_NUMBER_HIGH, 
            "high_true" : true_magic_high,
            "high_true-delta" : true_magic_high - DISTANCE_FROM_TRUE_MAGIC,
            "high_true+delta": true_magic_high + DISTANCE_FROM_TRUE_MAGIC,
        }

        processed_data[protocol][I_a_position][repetitions]["main_info"] = data[0]

        for i, key in enumerate(magic_high):
            extracted_data = extract_values_high(data, magic_high[key])
            meta_data = make_meta_high(data, extracted_data, magic_high[key])

            processed_data[protocol][I_a_position][repetitions][key] = {}

            processed_data[protocol][I_a_position][repetitions][key]["extracted"] = extracted_data
            processed_data[protocol][I_a_position][repetitions][key]["processed"] = meta_data

        true_magic_low = find_perfect_magic_number_low(data)

        magic_low = {
            "low_standard": MAGIC_NUMBER_LOW,
            "low_true": true_magic_low,
            "low_true-delta": true_magic_low - DISTANCE_FROM_TRUE_MAGIC,
            "low_true+delta": true_magic_low + DISTANCE_FROM_TRUE_MAGIC,
        }

        for i, key in enumerate(magic_low):
            extracted_data = extract_values_low(data, magic_low[key])
            meta_data = make_meta_low(data, extracted_data, magic_low[key])

            processed_data[protocol][I_a_position][repetitions][key] = {}

            processed_data[protocol][I_a_position][repetitions][key]["extracted"] = extracted_data
            processed_data[protocol][I_a_position][repetitions][key]["processed"] = meta_data

with open("processed_evolution.pkl", 'wb') as f:
    pickle.dump(processed_data, f)
