import os
import numpy as np
import re

PATH = "/eos/project-d/da-and-diffusion-studies/Diffusion_Studies/new_games_with_diffusion/data"

CHECK_PRESENCE = False

PARTIAL = True
PARTIAL_LIST = list(range(21))

def get_file_pars(file):
    elements = [float(x) for x in re.findall(r"\d+\.\d+", file)]
    return elements


def selector(file):
    I_max, I_step, fraction = get_file_pars(file)
    return (
        (I_step == 0.1) and
        (fraction == 0.5)
    )


files = list(filter(lambda x: "working" in x, os.listdir(PATH)))
all_files = os.listdir(PATH)

low_bound = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
high_bound = [np.inf, 1.0]

with open("file_list.txt", 'w') as out:
    for f in files:
        print(os.path.join(PATH, f))
        
        if not selector(f):
            print("Discarded")
            #continue
            PARTIAL = False
        else:
            print("Kept")
            PARTIAL = True

        for l in low_bound:
            #for bb, fb in [(True, True), (True, False), (False, True)]:
            for bb, fb in [(False, True)]:
                
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

                    if PARTIAL:
                        for P in PARTIAL_LIST:
                            out.write(
                                str(os.path.join(PATH, f)) + " " +
                                str(h) + " " +
                                str(l) + " " +
                                ("--backward " if bb else "--no-backward ") +
                                ("--forward " if fb else "--no-forward ") +
                                ("--partial " + str(P)) +
                                "\n"
                            )

