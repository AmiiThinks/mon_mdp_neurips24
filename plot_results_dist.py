# ruff: noqa: F403, F405
import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import importlib
import inspect
import sys

from src.plot_utils import *

sys.path.append("./configs/plots")

sns.set_context("paper")
# sns.set_style("whitegrid", {"legend.frameon": True})
sns.set_style("darkgrid", {"legend.frameon": True})
# plt.rcParams["axes.axisbelow"] = False
# plt.rcParams["axes.grid"] = False
plt.rcParams["grid.linestyle"] = "--"
# plt.rcParams["font.family"] = "DejaVu Sans Mono"
plt.rcParams["font.family"] = "Bree Serif"
font_size = 12
plt.rcParams["font.size"] = font_size

# ------------------------------------------------------------------------------
# Note that this script tries to use the Bree Serif font. To install it, check
# where matplotlib fonts are saved by running
#    from matplotlib.font_manager import findfont, FontProperties
#    print(findfont(FontProperties(family=["sans-serif"])))
#
# Then, download Bree Serif and install it there.
#
# Finally, delete matplotlib cache. To find it, run
#     matplotlib.get_cachedir()
# ------------------------------------------------------------------------------

def plot(folder):
    fig0, axs0 = make_subplots(nrows=1, ncols=1, width_per_plot=2.6, height_per_plot=2)
    fig1, axs1 = make_subplots(nrows=1, ncols=1, width_per_plot=2.6, height_per_plot=2)
    fig = [fig0, fig1]
    axs = [axs0[0][0], axs1[0][0]]

    for env, mon in list(
        itertools.product(sorted(env_to_label.keys()), sorted(mon_to_label.keys()))
    ):
        for ax in axs:
            ax.clear()
            ax.set_prop_cycle(None)

        nothing_to_plot = True
        print("\n\n>>>", env, mon)

        for cfg in benchmarks:
            (
                alg,
                q0_min,
                q0_max,
                q0_visit_min,
                q0_visit_max,
                eps_init,
                eps_min,
                beta_bar,
                label,
            ) = cfg

            data_test_return = []
            data_visited_r_sum = []
            seeds_completed = 0
            for seed in range(0, n_seeds):
                try:
                    filename = f"{alg}_{q0_min}_{q0_max}_{q0_visit_min}_{q0_visit_max}_{eps_init}_{eps_min}_{beta_bar}_{seed}.npz"
                    data = np.load(os.path.join(folder, env, mon, filename))
                    data_test_return.append(data["test/return"])
                    data_visited_r_sum.append(data["train/visited_r_sum"])
                    seeds_completed += 1
                except Exception as e:
                    print(e)
                    pass

            try:
                data_test_return = np.stack(data_test_return)
                data_visited_r_sum = np.stack(data_visited_r_sum)
            except Exception as e:
                print(e)
                pass

            n_bins = 100
            if "RiverSwim" in env:
                bins = np.linspace(-0.02, 22.0, n_bins)
            else:
                bins = np.linspace(-0.02, 1.0, n_bins)

            if len(data_test_return) > 0:
                args = {
                    "label": label,
                    "ls": "-",
                    "color": alg_to_color[alg],
                    "fill": True,
                    "linewidth": 0.5,
                }
                # sns.kdeplot(data_test_return[:, -1], ax=axs[0], **args)
                # sns.kdeplot(data_visited_r_sum[:, -1], ax=axs[1], **args)
                sns.kdeplot(data_test_return.mean(-1), ax=axs[0], **args)
                sns.kdeplot(data_visited_r_sum.mean(-1), ax=axs[1], **args)
                nothing_to_plot = False

        if nothing_to_plot:
            continue

        for i in range(len(axs)):
            axs[i].tick_params(axis="x", labelsize=font_size - 2, pad=-4)
            axs[i].tick_params(axis="y", labelsize=font_size - 2, pad=y_tick_pad + 17)
            if "FullMonitor" not in mon:
                axs[i].set_ylabel("")

        if "FullMonitor" not in mon:
            axs[0].set_ylabel("Expected Return AUC")
            axs[1].set_ylabel("Observed Reward AUC")

        plt.draw()

        def save(fig, name):
            plt.figure(fig)
            savepath = os.path.join(folder, savedir, name)
            os.makedirs(savepath, exist_ok=True)
            savename = os.path.join(env, mon, "").replace("\\", "_").replace("/", "_")
            savename = os.path.join(savepath, savename + ".png")
            plt.savefig(savename, bbox_inches="tight", pad_inches=0, dpi=1500)

        save(fig[0], "return")
        save(fig[1], "visited_r_sum")

        print("--------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config")
    parser.add_argument("-f", "--folder")
    args = parser.parse_args()

    # https://stackoverflow.com/a/77350187/754136
    # inject config variables into the global namespace
    cfgmod = importlib.import_module(inspect.getmodulename(args.config))
    dicts = {k: v for k, v in inspect.getmembers(cfgmod) if not k.startswith("_")}
    globals().update(**dicts)

    plot(args.folder)
