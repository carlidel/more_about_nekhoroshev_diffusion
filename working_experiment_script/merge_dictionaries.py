import os
import pickle
from tqdm import tqdm

PATH = "/eos/project-d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"

def merge(a, b, path=None):
    "merges b into a"
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

full_dict = {}

files = list(filter(lambda x: "FIT_u" in x[0:6], os.listdir(PATH)))

for f in tqdm(files):
    print(f)
    with open(os.path.join(PATH, f), 'rb') as input:
        partial_dict = pickle.load(input)
    print("merge")
    full_dict = merge(full_dict, partial_dict)
    print("done")

with open("full_dict.pkl", 'wb') as f:
    pickle.dump(full_dict, f)
