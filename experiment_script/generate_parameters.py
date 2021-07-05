import json

base_parameter = {
    "I_star": 20.0,
    "k": 0.33,

    "step_size": 0.01,
    "cn_sampling": 10000,
    "dt_sampling": 1000,

    "steps_in_I": 50,
    "relax_samples": 500000,

    "relaxing_samples": 500000,
    "it_per_sample": 2
}

damping_distance = 0.05

repetition_parameters = []

for i in [50, 20, 10, 5, 4, 2, 1]:
    repetition_parameters.append(
        {
            "sub_name": "r_{}".format(i),
            "repetitions": i,
            "samples": 500000 // i,
        }
    )

plugin_parameters = [
    {
        "name": "lowlow_I_a_2",
        "I_max": 2.0,
    },
    
    {
        "name": "low_I_a_2",
        "I_max": 4.0,
    },
    
    {
        "name": "avg_I_a_2",
        "I_max": 6.0,
    },

    {
        "name": "mid_I_a_2",
        "I_max": 10.0,
    },

    {
        "name": "high_I_a_2",
        "I_max": 15.0,
    },

    {
        "name": "higher_I_a_2",
        "I_max": 18.0,
    },

    {
        "name": "same_I_a_2",
        "I_max": 20.0,
    },

    {
        "name": "ultra_I_a_2",
        "I_max": 25.0,
    },

    {
        "name": "over_I_a_2",
        "I_max": 35.0,
    },
]

for d in plugin_parameters:
    d["I_damping"] = d["I_max"] - damping_distance

with open("base_experiment.sub", 'r') as f:
    base_experiment = f.read()

base_experiment += "\n\n"

block = (
    "immovable=--movable\n" +
    "protocol=0\n" +
    "queue\n" +
    "protocol=1\n" +
    "queue\n" +
    "protocol=2\n" +
    "queue\n\n" +
    "immovable=--immovable\n" +
    "protocol=0\n" +
    "queue\n" +
    "protocol=1\n" +
    "queue\n" +
    "protocol=2\n" +
    "queue\n\n"
)

for d1 in plugin_parameters:
    for d2 in repetition_parameters:
        d = {**base_parameter, **d1, **d2}
        namefile = "par_" + d["name"] + "_" + d["sub_name"] + ".json"
        with open(namefile, 'w') as f:
            json.dump(d, f, indent=4)

        base_experiment += "file=" + namefile + "\n\n" + block

with open("execute_experiment_v2.sub", 'w') as f:
    f.write(base_experiment)
