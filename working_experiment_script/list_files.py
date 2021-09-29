import os

PATH = "/eos/project-d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"

files = list(filter(lambda x: "working" in x, os.listdir(PATH)))

with open("file_list.txt", 'w') as out:
    for f in files:
        print(os.path.join(PATH, f))
        out.write(str(os.path.join(PATH, f)) + "\n")

