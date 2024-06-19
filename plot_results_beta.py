# ruff: noqa: F403, F405
import os
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
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
plt.rcParams["axes.axisbelow"] = False
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
    fig, axs = make_subplots(nrows=1, ncols=1, width_per_plot=2.6, height_per_plot=2)
    fig_one, axs_one = make_subplots(nrows=1, ncols=1, width_per_plot=2.6, height_per_plot=2)
    fig = [fig, fig_one]
    axs = [axs[0][0], axs_one[0][0]]

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
            if alg != "q_visit":
                continue

            data_beta = []
            seeds_completed = 0
            for seed in range(0, n_seeds):
                try:
                    filename = f"{alg}_{q0_min}_{q0_max}_{q0_visit_min}_{q0_visit_max}_{eps_init}_{eps_min}_{beta_bar}_{seed}.npz"
                    data = np.load(os.path.join(folder, env, mon, filename))
                    data_beta.append(data["train/beta"])
                    log_frequency = data["training_steps"] / len(data["train/beta"])
                    seeds_completed += 1
                except Exception as e:
                    print(e)
                    pass

            try:
                data_beta = np.stack(data_beta)
                data_beta[np.isinf(data_beta)] = 10
            except Exception as e:
                print(e)
                pass

            if len(data_beta) > 0:
                args = {
                    "label": label,
                    "lw": 2,
                    "ls": "-",
                    "color": alg_to_color[alg],
                    "marker": "",
                    "markersize": 1,
                    "markevery": 10,
                }
                extra_args = {
                    "smoothing_window": smoothing_window,
                    "stepsize": log_frequency,
                }
                steps = log_frequency * np.arange(data_beta.shape[1])
                beta_bar = np.zeros_like(steps) + float(beta_bar)
                error_shade_plot(axs[0], data_beta, **extra_args, **args)
                axs[1].plot(steps, data_beta[0], **args)
                args["color"] = "k"
                axs[0].plot(steps, beta_bar, **args)
                axs[1].plot(steps, beta_bar, **args)
                nothing_to_plot = False

        if nothing_to_plot:
            continue

        xlim = data["training_steps"]

        for i in range(len(axs)):
            axs[i].yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            axs[i].set_yscale("log")
            # axs[i].set_ylim([None, 10.1])

            axs[i].tick_params(axis="x", labelsize=font_size - 2, pad=-4)
            axs[i].tick_params(axis="y", labelsize=font_size - 2, pad=y_tick_pad + 17)

            axs[i].set_xlim([-xlim // 200, xlim + xlim // 200])  # 2% margin/padding looks nice
            axs[i].ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
            # axs[i].set_xlabel("Training Steps (1e3)", fontsize=font_size, labelpad=-23, loc="right")
            axs[i].xaxis.offsetText.set_visible(False)  # hide the exp notation
            axs[i].xaxis.set_ticks(np.linspace(0, xlim, 3, dtype=np.int32))

        axs[1].xaxis.set_ticks(np.linspace(0, xlim, 11, dtype=np.int32))

        plt.draw()

        def save(fig, name):
            plt.figure(fig)
            savepath = os.path.join(folder, savedir, name)
            os.makedirs(savepath, exist_ok=True)
            savename = os.path.join(env, mon, "").replace("\\", "_").replace("/", "_")
            savename = os.path.join(savepath, savename + ".png")
            plt.savefig(savename, bbox_inches="tight", pad_inches=0, dpi=1500)

        save(fig[0], "beta")
        save(fig[1], "beta_one")

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
