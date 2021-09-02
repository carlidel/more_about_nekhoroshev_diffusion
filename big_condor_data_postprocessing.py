import numpy as np
import scipy
import scipy.integrate
import scipy.interpolate
import scipy.optimize
from tqdm import tqdm
import pickle
import os
import re
import subprocess
import argparse

import nekhoroshev_tools as nt

DATA_PATH = "/eos/project/d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"
OUT_PATH = "/afs/cern.ch/work/c/camontan/public/more_about_nekhoroshev_diffusion/data"

DATA_PATH_LOCAL = "/home/camontan/moving_barrier_data"
OUT_PATH_LOCAL = "/home/camontan/moving_barrier_data"

MAGIC_NUMBER_HIGH = 1.01
MAGIC_NUMBER_LOW = 0.99
DISTANCE_FROM_TRUE_MAGIC = 0.005

def interpolation_system(data):
    low_points_time = []
    low_points_value = []
    high_points_time = []
    high_points_value = []

    low_points_value_ana = []
    high_points_value_ana = []
    for i, d in enumerate(data[2]):
        if d["I_max_before"] == d["I_max_after"]:
            low_points_time.append(d["t_absolute"][-1])
            high_points_time.append(d["t_absolute"][-1])

            low_points_value.append(d["current"][-1])
            high_points_value.append(d["current"][-1])
            
            low_points_value_ana.append(d["current_ana"][-1])
            high_points_value_ana.append(d["current_ana"][-1])

        elif d["I_max_before"] < d["I_max_after"]:
            low_points_time.append(d["t_absolute"][-1])
            low_points_value.append(d["current"][-1])
            low_points_value_ana.append(d["current_ana"][-1])

        elif d["I_max_before"] > d["I_max_after"]:
            high_points_time.append(d["t_absolute"][-1])
            high_points_value.append(d["current"][-1])
            high_points_value_ana.append(d["current_ana"][-1])

        if i == len(data[2]) - 1:
            if d["I_max_before"] < d["I_max_after"]:
                high_points_time.append(d["t_absolute"][-1])
                high_points_value.append(d["current"][-1])
                high_points_value_ana.append(d["current_ana"][-1])
            else:
                low_points_time.append(d["t_absolute"][-1])
                low_points_value.append(d["current"][-1])
                low_points_value_ana.append(d["current_ana"][-1])

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

    low_points_f_ana = scipy.interpolate.interp1d(
        low_points_time,
        low_points_value_ana,
        kind='cubic'
    )

    high_points_f_ana = scipy.interpolate.interp1d(
        high_points_time,
        high_points_value_ana,
        kind='cubic'
    )

    def mid_points_ana(t):
        return (high_points_f_ana(t) + low_points_f_ana(t)) / 2

    return(
        low_points_f, mid_points, high_points_f,
        low_points_f_ana, mid_points_ana, high_points_f_ana
    )


def add_normed_current(data, imm_data):
    low_points_f, mid_points, high_points_f, low_points_f_ana, mid_points_ana, high_points_f_ana = interpolation_system(
        data)
    for i, d in enumerate(data[2]):
        if d["I_max_before"] != d["I_max_after"]:
            d["n_current"] = d["current"]/mid_points(d["t_absolute"])
            d["n_current_low"] = d["current"]/low_points_f(d["t_absolute"])
            d["n_current_high"] = d["current"]/high_points_f(d["t_absolute"])
            # True normalized current
            d["n_current_true"] = d["current"] / imm_data[2][i]["current"]

            d["n_current_ana"] = d["current_ana"] / \
                mid_points_ana(d["t_absolute"])
            d["n_current_low_ana"] = d["current_ana"] / \
                low_points_f_ana(d["t_absolute"])
            d["n_current_high_ana"] = d["current_ana"] / \
                high_points_f_ana(d["t_absolute"])
            # True normalized current
            d["n_current_true_ana"] = d["current_ana"] / \
                imm_data[2][i]["current_ana"]
    return data


def gather_avg_currents(data, kind="high"):
    start_points = set()
    for d in data[2]:
        if kind == "low":
            if d["I_max_before"] < d["I_max_after"]:
                start_points.add(d["I_max_before"])
        if kind == "high":
            if d["I_max_before"] > d["I_max_after"]:
                start_points.add(d["I_max_before"])
    start_points = list(sorted(list(start_points)))

    tmp_data = {
        i: {
            j: {
                "mid": [],
                "high": [],
                "low": [],
                "true": [],
            } for j in ["ana", "num"]
        }
        for i in start_points
    }
    cur_data = {
        i: {
            j:{
                "mid": {"avg": [], "std": []},
                "high": {"avg": [], "std": []},
                "low": {"avg": [], "std": []},
                "true": {"avg": [], "std": []},
                "time": []
            } for j in ["ana", "num"]
        }
        for i in start_points
    }

    for d in data[2]:
        flag = False
        if kind == "high":
            flag = d["I_max_before"] > d["I_max_after"]
        if kind == "low":
            flag = d["I_max_before"] < d["I_max_after"]
        if "n_current" in d and flag:
            tmp_data[d["I_max_before"]]["num"]["mid"].append(d["n_current"])
            tmp_data[d["I_max_before"]]["num"]["high"].append(d["n_current_high"])
            tmp_data[d["I_max_before"]]["num"]["low"].append(d["n_current_low"])
            tmp_data[d["I_max_before"]]["num"]["true"].append(d["n_current_true"])

            cur_data[d["I_max_before"]]["num"]["time"] = d["t_relative"]
        
            tmp_data[d["I_max_before"]]["ana"]["mid"].append(d["n_current_ana"])
            tmp_data[d["I_max_before"]]["ana"]["high"].append(d["n_current_high_ana"])
            tmp_data[d["I_max_before"]]["ana"]["low"].append(d["n_current_low_ana"])
            tmp_data[d["I_max_before"]]["ana"]["true"].append(d["n_current_true_ana"])

            cur_data[d["I_max_before"]]["ana"]["time"] = d["t_relative"]

    for key in tmp_data:
        cur_data[key]["num"]["mid"]["avg"] = np.mean(
            tmp_data[key]["num"]["mid"], axis=0)
        cur_data[key]["num"]["high"]["avg"] = np.mean(
            tmp_data[key]["num"]["high"], axis=0)
        cur_data[key]["num"]["low"]["avg"] = np.mean(
            tmp_data[key]["num"]["low"], axis=0)
        cur_data[key]["num"]["true"]["avg"] = np.mean(
            tmp_data[key]["num"]["true"], axis=0)

        cur_data[key]["num"]["mid"]["std"] = np.std(
            tmp_data[key]["num"]["mid"], axis=0)
        cur_data[key]["num"]["high"]["std"] = np.std(
            tmp_data[key]["num"]["high"], axis=0)
        cur_data[key]["num"]["low"]["std"] = np.std(
            tmp_data[key]["num"]["low"], axis=0)
        cur_data[key]["num"]["true"]["std"] = np.std(
            tmp_data[key]["num"]["true"], axis=0)

        cur_data[key]["ana"]["mid"]["avg"] = np.mean(
            tmp_data[key]["ana"]["mid"], axis=0)
        cur_data[key]["ana"]["high"]["avg"] = np.mean(
            tmp_data[key]["ana"]["high"], axis=0)
        cur_data[key]["ana"]["low"]["avg"] = np.mean(
            tmp_data[key]["ana"]["low"], axis=0)
        cur_data[key]["ana"]["true"]["avg"] = np.mean(
            tmp_data[key]["ana"]["true"], axis=0)

        cur_data[key]["ana"]["mid"]["std"] = np.std(
            tmp_data[key]["ana"]["mid"], axis=0)
        cur_data[key]["ana"]["high"]["std"] = np.std(
            tmp_data[key]["ana"]["high"], axis=0)
        cur_data[key]["ana"]["low"]["std"] = np.std(
            tmp_data[key]["ana"]["low"], axis=0)
        cur_data[key]["ana"]["true"]["std"] = np.std(
            tmp_data[key]["ana"]["true"], axis=0)

    return cur_data


def extract_values(data, magic_number, kind="high"):
    start_points = set()
    for d in data[2]:
        if kind == "high":
            if d["I_max_before"] > d["I_max_after"]:
                start_points.add(d["I_max_before"])
        if kind == "low":
            if d["I_max_before"] < d["I_max_after"]:
                start_points.add(d["I_max_before"])
    start_points = list(sorted(list(start_points)))
    post_data = {
        i: {
            j: {
                "mid": [],
                "low": [],
                "high": [],
                "true": [],
            } for j in ["ana", "num"]
        }
        for i in start_points
    }
    for d in data[2]:
        flag = False
        if kind == "high":
            flag = d["I_max_before"] > d["I_max_after"]
        if kind == "low":
            flag = d["I_max_before"] < d["I_max_after"]
        if "n_current" in d and flag:
            temp = np.argmin(np.absolute(
                magic_number - d["n_current"]))
            post_data[d["I_max_before"]]["num"]["mid"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number - d["n_current_low"]))
            post_data[d["I_max_before"]]["num"]["low"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number - d["n_current_high"]))
            post_data[d["I_max_before"]]["num"]["high"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number - d["n_current_true"]))
            post_data[d["I_max_before"]]["num"]["true"].append(
                d["t_relative"][temp]
            )

            temp = np.argmin(np.absolute(
                magic_number - d["n_current_ana"]))
            post_data[d["I_max_before"]]["ana"]["mid"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number - d["n_current_low_ana"]))
            post_data[d["I_max_before"]]["ana"]["low"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number - d["n_current_high_ana"]))
            post_data[d["I_max_before"]]["ana"]["high"].append(
                d["t_relative"][temp]
            )
            temp = np.argmin(np.absolute(
                magic_number - d["n_current_true_ana"]))
            post_data[d["I_max_before"]]["ana"]["true"].append(
                d["t_relative"][temp]
            )
    return post_data


def make_meta(data, point_data, magic_value, kind="high"):
    val = []

    avg = []
    std = []
    avg_low = []
    std_low = []
    avg_high = []
    std_high = []
    avg_true = []
    std_true = []

    avg_ana = []
    std_ana = []
    avg_low_ana = []
    std_low_ana = []
    avg_high_ana = []
    std_high_ana = []
    avg_true_ana = []
    std_true_ana = []

    afpt = []
    for key in point_data:
        val.append(key)

        avg.append(np.average(point_data[key]["num"]["mid"]))
        std.append(np.std(point_data[key]["num"]["mid"]))
        avg_low.append(np.average(point_data[key]["num"]["low"]))
        std_low.append(np.std(point_data[key]["num"]["low"]))
        avg_high.append(np.average(point_data[key]["num"]["high"]))
        std_high.append(np.std(point_data[key]["num"]["high"]))
        avg_true.append(np.average(point_data[key]["num"]["true"]))
        std_true.append(np.std(point_data[key]["num"]["true"]))

        avg_ana.append(np.average(point_data[key]["ana"]["mid"]))
        std_ana.append(np.std(point_data[key]["ana"]["mid"]))
        avg_low_ana.append(np.average(point_data[key]["ana"]["low"]))
        std_low_ana.append(np.std(point_data[key]["ana"]["low"]))
        avg_high_ana.append(np.average(point_data[key]["ana"]["high"]))
        std_high_ana.append(np.std(point_data[key]["ana"]["high"]))
        avg_true_ana.append(np.average(point_data[key]["ana"]["true"]))
        std_true_ana.append(np.std(point_data[key]["ana"]["true"]))

        afpt.append(
            nt.afpt(
                key if kind == "high" else key - data[0]["step_size"],
                key + data[0]["step_size"] if kind == "low" else key,
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
        "avg_ana": np.asarray(avg_ana),
        "std_ana": np.asarray(std_ana),
        "avg_low_ana": np.asarray(avg_low_ana),
        "std_low_ana": np.asarray(std_low_ana),
        "avg_high_ana": np.asarray(avg_high_ana),
        "std_high_ana": np.asarray(std_high_ana),
        "avg_true_ana": np.asarray(avg_true_ana),
        "std_true_ana": np.asarray(std_true_ana),
        "afpt": np.asarray(afpt),
    }


def test_magic_number_high(val, data):
    high_data = extract_values(data, val, kind="high")
    meta_high_data = make_meta(data, high_data, val, kind="high")
    result = np.sum(np.absolute(np.log10(
        meta_high_data["avg"]) - np.log10(np.absolute(meta_high_data["afpt"]))))
    return(result)


def test_magic_number_low(val, data):
    low_data = extract_values(data, val, kind="low")
    meta_low_data = make_meta(data, low_data, val, kind="low")
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
        0.98,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-l',
        '--local',
        action='store_true',
        help='perform the thing locally and not on htcondor'
    )
    parser.add_argument(
        '-a',
        '--averages',
        action='store_true',
        help='perform the averages'
    )
    args = parser.parse_args()

    if args.local:
        DATA_PATH = DATA_PATH_LOCAL
        OUT_PATH = OUT_PATH_LOCAL

    processed_data = {}

    files = list(sorted(os.listdir(DATA_PATH)))

    for f in tqdm(list(filter(lambda f:
                              "I_a_4" in f
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
        
        try:
            with open(os.path.join(DATA_PATH, f_imm), 'rb') as file:
                imm_data = pickle.load(file)
        except Exception:
            print("No immovable file! Skipping...")
            continue

        data = add_normed_current(data, imm_data)

        if protocol not in processed_data:
            processed_data[protocol] = {}

        if I_a_position not in processed_data[protocol]:
            processed_data[protocol][I_a_position] = {}

        if repetitions not in processed_data[protocol][I_a_position]:
            processed_data[protocol][I_a_position][repetitions] = {}

        true_magic_high = find_perfect_magic_number_high(data)

        magic_high = {
            "high_standard": MAGIC_NUMBER_HIGH,
            "high_true": true_magic_high,
            "high_true-delta": true_magic_high - DISTANCE_FROM_TRUE_MAGIC,
            "high_true+delta": true_magic_high + DISTANCE_FROM_TRUE_MAGIC,
        }

        processed_data[protocol][I_a_position][repetitions]["main_info"] = data[0]

        for i, key in tqdm(list(enumerate(magic_high))):
            extracted_data = extract_values(data, magic_high[key], kind="high")
            meta_data = make_meta(data, extracted_data, magic_high[key], kind="high")

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
            extracted_data = extract_values(data, magic_low[key], kind="low")
            meta_data = make_meta(data, extracted_data, magic_low[key], kind="low")

            processed_data[protocol][I_a_position][repetitions][key] = {}

            processed_data[protocol][I_a_position][repetitions][key]["extracted"] = extracted_data
            processed_data[protocol][I_a_position][repetitions][key]["processed"] = meta_data

        if args.averages:
            print("Extracting the averages...")

            avg_data_high = gather_avg_currents(data, kind="high")
            avg_data_low = gather_avg_currents(data, kind="low")

            print("Saving high...")
            with open(os.path.join(OUT_PATH, f[:-4] + "_high_avg.pkl"), 'wb') as file:
                pickle.dump(avg_data_high, file)

            if not args.local:
                subprocess.run([
                    "eos", "cp",
                    os.path.join(OUT_PATH, f[:-4] + "_high_avg.pkl"),
                    "/eos/project/d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"
                ])
                os.remove(os.path.join(OUT_PATH, f[:-4] + "_high_avg.pkl"))

            print("Saving low...")
            with open(os.path.join(OUT_PATH, f[:-4] + "_low_avg.pkl"), 'wb') as file:
                pickle.dump(avg_data_low, file)

            if not args.local:
                subprocess.run([
                    "eos", "cp",
                    os.path.join(OUT_PATH, f[:-4] + "_low_avg.pkl"),
                    "/eos/project/d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"
                ])
                os.remove(os.path.join(OUT_PATH, f[:-4] + "_low_avg.pkl"))

    with open(os.path.join(OUT_PATH, "processed_evolution_tiptap_only.pkl"), 'wb') as file:
        print("Saving...")
        pickle.dump(processed_data, file)
    
    if not args.local:
        subprocess.run([
            "eos", "cp", 
            os.path.join(OUT_PATH, "processed_evolution_tiptap_only.pkl"),
            "/eos/project/d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"
        ])
        os.remove(os.path.join(OUT_PATH, "processed_evolution_tiptap_only.pkl"))


if __name__ == "__main__":
    main()
    
