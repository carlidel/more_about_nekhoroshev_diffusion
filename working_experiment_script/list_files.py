import os
import numpy as np

PATH = "/eos/project-d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"

CHECK_PRESENCE = True

files = list(filter(lambda x: "working" in x, os.listdir(PATH)))
all_files = os.listdir(PATH)

low_bound = [0.5, 0.1, 0.01, 0.001]
high_bound = [np.inf, 1.0]

with open("file_list.txt", 'w') as out:
    for f in files:
        print(os.path.join(PATH, f))
        for l in low_bound:
            for bb, fb in [(True, True), (True, False), (False, True)]:
                
                if fb and bb:
                    method = "all"
                elif fb and not bb:
                    method = "forward_only"
                elif bb and not fb:
                    method = "backward_only"
                else:
                    method = "error"
                    assert(False)

                if bb:
                    for h in high_bound:
                        if CHECK_PRESENCE:
                            outname = "FIT_ub_{}_lb_{}_md_{}_".format(
                                h, l, method) + f.replace("working_experiment_", "")
                            if outname in all_files:
                                flag = False
                            else:
                                flag = True
                        else:
                            flag = True
                        
                        if flag:
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

                    if CHECK_PRESENCE:
                        outname = "FIT_ub_{}_lb_{}_md_{}_".format(h, l, method) + f.replace("working_experiment_", "")
                        if outname in all_files:
                            flag = False
                        else:
                            flag = True
                    else:
                        flag = True

                    if flag:
                        out.write(
                            str(os.path.join(PATH, f)) + " " +
                            str(h) + " " +
                            str(l) + " " +
                            ("--backward " if bb else "--no-backward ") +
                            ("--forward " if fb else "--no-forward ") +
                            "\n"
                        )

