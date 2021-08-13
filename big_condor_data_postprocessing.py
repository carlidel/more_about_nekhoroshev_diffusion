import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.optimize
from tqdm import tqdm
import pickle
import os
import re

import nekhoroshev_tools as nt

DATA_PATH = "/eos/project/d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"
OUT_PATH = "/afs/cern.ch/work/c/camontan/public/more_about_nekhoroshev_diffusion/data"

#DATA_PATH = "/home/camontan/moving_barrier_data"
#OUT_PATH = "."

MAGIC_NUMBER_HIGH = 1.02
MAGIC_NUMBER_LOW = 0.98
DISTANCE_FROM_TRUE_MAGIC = 0.005

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


def add_normed_current(data, imm_data):
    low_points_f, mid_points, high_points_f = interpolation_system(data)
    for i, d in enumerate(data[2]):
        if d["I_max_before"] != d["I_max_after"]:
            d["n_current"] = d["current"]/mid_points(d["t_absolute"])
            d["n_current_low"] = d["current"]/low_points_f(d["t_absolute"])
            d["n_current_high"] = d["current"]/high_points_f(d["t_absolute"])
            # True normalized current
            d["n_current_true"] = d["current"] / imm_data[2][i]["current"]
    return data


def gather_avg_currents_high(data):
    start_points = set()
    for d in data[2]:
        if d["I_max_before"] > d["I_max_after"]:
            start_points.add(d["I_max_before"])
    start_points = list(sorted(list(start_points)))

    tmp_data = {
        i: {
            "mid": [],
            "high": [],
            "low": [],
            "true": [],
        }
        for i in start_points
    }
    cur_data = {
        i: {
            "mid": {"avg": [], "std": []},
            "high": {"avg": [], "std": []},
            "low": {"avg": [], "std": []},
            "true": {"avg": [], "std": []},
            "time": []
        }
        for i in start_points
    }

    for d in data[2]:
        if "n_current" in d and d["I_max_before"] > d["I_max_after"]:
            tmp_data[d["I_max_before"]]["mid"].append(d["n_current"])
            tmp_data[d["I_max_before"]]["high"].append(d["n_current_high"])
            tmp_data[d["I_max_before"]]["low"].append(d["n_current_low"])
            tmp_data[d["I_max_before"]]["true"].append(d["n_current_true"])

            cur_data[d["I_max_before"]]["time"] = d["t_relative"]

    for key in tmp_data:
        cur_data[key]["mid"]["avg"] = np.mean(tmp_data[key]["mid"], axis=0)
        cur_data[key]["high"]["avg"] = np.mean(tmp_data[key]["high"], axis=0)
        cur_data[key]["low"]["avg"] = np.mean(tmp_data[key]["low"], axis=0)
        cur_data[key]["true"]["avg"] = np.mean(tmp_data[key]["true"], axis=0)

        cur_data[key]["mid"]["std"] = np.std(tmp_data[key]["mid"], axis=0)
        cur_data[key]["high"]["std"] = np.std(tmp_data[key]["high"], axis=0)
        cur_data[key]["low"]["std"] = np.std(tmp_data[key]["low"], axis=0)
        cur_data[key]["true"]["std"] = np.std(tmp_data[key]["true"], axis=0)

    return cur_data


def gather_avg_currents_low(data):
    start_points = set()
    for d in data[2]:
        if d["I_max_before"] < d["I_max_after"]:
            start_points.add(d["I_max_before"])
    start_points = list(sorted(list(start_points)))

    tmp_data = {
        i: {
            "mid": [],
            "high": [],
            "low": [],
            "true": [],
        }
        for i in start_points
    }
    cur_data = {
        i: {
            "mid": {"avg": [], "std": []},
            "high": {"avg": [], "std": []},
            "low": {"avg": [], "std": []},
            "true": {"avg": [], "std": []},
            "time": []
        }
        for i in start_points
    }

    for d in data[2]:
        if "n_current" in d and d["I_max_before"] < d["I_max_after"]:
            tmp_data[d["I_max_before"]]["mid"].append(d["n_current"])
            tmp_data[d["I_max_before"]]["high"].append(d["n_current_high"])
            tmp_data[d["I_max_before"]]["low"].append(d["n_current_low"])
            tmp_data[d["I_max_before"]]["true"].append(d["n_current_true"])
    
            cur_data[d["I_max_before"]]["time"] = d["t_relative"]


    for key in tmp_data:
        cur_data[key]["mid"]["avg"] = np.mean(tmp_data[key]["mid"], axis=0)
        cur_data[key]["high"]["avg"] = np.mean(tmp_data[key]["high"], axis=0)
        cur_data[key]["low"]["avg"] = np.mean(tmp_data[key]["low"], axis=0)
        cur_data[key]["true"]["avg"] = np.mean(tmp_data[key]["true"], axis=0)

        cur_data[key]["mid"]["std"] = np.std(tmp_data[key]["mid"], axis=0)
        cur_data[key]["high"]["std"] = np.std(tmp_data[key]["high"], axis=0)
        cur_data[key]["low"]["std"] = np.std(tmp_data[key]["low"], axis=0)
        cur_data[key]["true"]["std"] = np.std(tmp_data[key]["true"], axis=0)

    return cur_data


def extract_values_high(data, magic_number_high):
    start_points = set()
    for d in data[2]:
        if d["I_max_before"] > d["I_max_after"]:
            start_points.add(d["I_max_before"])
    start_points = list(sorted(list(start_points)))
    high_data = {
        i: {
            "mid": [],
            "low": [],
            "high": [],
            "true": [],
        }
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
            temp = np.argmin(np.absolute(
                magic_number_high - d["n_current_true"]))
            high_data[d["I_max_before"]]["true"].append(
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
        i: {
            "mid": [],
            "low": [],
            "high": [],
            "true": []
        }
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
            temp = np.argmin(np.absolute(
                magic_number_low - d["n_current_true"]))
            low_data[d["I_max_before"]]["true"].append(
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
    avg_true = []
    std_true = []
    afpt = []
    for key in point_data:
        val.append(key)

        avg.append(np.average(point_data[key]["mid"]))
        std.append(np.std(point_data[key]["mid"]))

        avg_low.append(np.average(point_data[key]["low"]))
        std_low.append(np.std(point_data[key]["low"]))

        avg_high.append(np.average(point_data[key]["high"]))
        std_high.append(np.std(point_data[key]["high"]))

        avg_true.append(np.average(point_data[key]["true"]))
        std_true.append(np.std(point_data[key]["true"]))

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
        "avg_true": np.asarray(avg_true),
        "std_true": np.asarray(std_true),
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
    avg_true = []
    std_true = []
    afpt = []
    for key in point_data:
        val.append(key)

        avg.append(np.average(point_data[key]["mid"]))
        std.append(np.std(point_data[key]["mid"]))

        avg_low.append(np.average(point_data[key]["low"]))
        std_low.append(np.std(point_data[key]["low"]))

        avg_high.append(np.average(point_data[key]["high"]))
        std_high.append(np.std(point_data[key]["high"]))

        avg_true.append(np.average(point_data[key]["true"]))
        std_true.append(np.std(point_data[key]["true"]))

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
        "avg_true": np.asarray(avg_true),
        "std_true": np.asarray(std_true),
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
    
    for f in tqdm(list(filter(lambda f: 
            "I_a_3" in f 
            and "immovable" not in f 
            and "high_avg" not in f 
            and "low_avg" not in f
            and "steady_tip-tap" in f, files))):
        print("Processing", f)
        f_imm = re.sub("standard", "immovable", f)
        print("Immovable file is", f_imm)

        version, protocol, I_a_position, repetitions = decompose_filename(f)

        with open(os.path.join(DATA_PATH, f), 'rb') as file:
            data = pickle.load(file)      
        with open(os.path.join(DATA_PATH, f_imm), 'rb') as file:
            imm_data = pickle.load(file)

        data = add_normed_current(data, imm_data)

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

        for i, key in tqdm(list(enumerate(magic_high))):
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

        for i, key in tqdm(list(enumerate(magic_low))):
            extracted_data = extract_values_low(data, magic_low[key])
            meta_data = make_meta_low(data, extracted_data, magic_low[key])

            processed_data[protocol][I_a_position][repetitions][key] = {}

            processed_data[protocol][I_a_position][repetitions][key]["extracted"] = extracted_data
            processed_data[protocol][I_a_position][repetitions][key]["processed"] = meta_data

        # print("Extracting the averages...")

        # avg_data_high = gather_avg_currents_high(data)
        # avg_data_low = gather_avg_currents_low(data)

        # print("Saving high...")
        # with open(os.path.join(OUT_PATH, f[:-4] + "_high_avg.pkl"), 'wb') as file:
        #     pickle.dump(avg_data_high, file)

        # print("Saving low...")
        # with open(os.path.join(OUT_PATH, f[:-4] + "_low_avg.pkl"), 'wb') as file:
        #     pickle.dump(avg_data_low, file)

    with open(os.path.join(OUT_PATH, "processed_evolution_tiptap_only.pkl"), 'wb') as file:
        print("Saving...")
        pickle.dump(processed_data, file)
