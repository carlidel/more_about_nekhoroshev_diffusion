import os
import numpy as np

PATH = "/eos/project-d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"

files = list(filter(lambda x: "working" in x, os.listdir(PATH)))

low_bound = [0.5, 0.1, 0.01, 0.001]
high_bound = [np.inf, 1.0]

with open("file_list.txt", 'w') as out:
    for f in files:
        print(os.path.join(PATH, f))
        for l in low_bound:
            for bb, fb in [(True, True), (True, False), (False, True)]:
                if bb:
                    for h in high_bound:
                        out.write(
                            str(os.path.join(PATH, f)) + " " +
                            str(h) + " " +
                            str(l) + " " +
                            ("--backward " if bb else "--no-backward ") +
                            ("--forward " if fb else "--no-forward ") +
                            "\n"
                        )
                else:
                    h=np.inf
                    out.write(
                        str(os.path.join(PATH, f)) + " " +
                        str(h) + " " +
                        str(l) + " " +
                        ("--backward " if bb else "--no-backward ") +
                        ("--forward " if fb else "--no-forward ") +
                        "\n"
                    )

