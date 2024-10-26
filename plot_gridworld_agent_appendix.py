# ruff: noqa: F403, F405
from matplotlib import pyplot as plt
plt.rcParams["font.family"] = "Bree Serif"
plt.rcParams["font.size"] = 14
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os

# YOU MUST REPLACE plot_agent IN plot_gridworld_agent.py WITH THIS ONE
def plot_agent(actor, critic, savepath=""):
    def draw(arr, width, height, format, filepath, vmin, vmax, colorbar):
        fig.set_figwidth(width)
        fig.set_figheight(height)
        axs.clear()
        # im = axs.imshow(arr, vmin=vmin, vmax=vmax, cmap="rocket")
        im = axs.imshow(arr, vmin=vmin, vmax=vmax, cmap="magma")
        axs.set_xticks([0, 1, 2, 3, 4, 5])
        axs.set_yticks([0, 1], ["L", "R"])

        if colorbar:
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="5%", pad=0.25)
            # m = (vmin + vmax) / 2
            # ticks = [np.ceil(vmin) + m * 0.1, np.floor(vmax) - m * 0.1]
            # ticklabels = [np.ceil(vmin), np.floor(vmax)]
            ticks = [vmin, vmax]
            cb = plt.colorbar(im, cax=cax, format=format, ticks=ticks)

        # for (j, i), label in np.ndenumerate(arr):
        #     if not np.isnan(label):
        #         axs.text(i, j, format.format(label), ha='center', va='center')

        axs.set_title("OFF" if "off" in filepath else "ON")
        plt.draw()
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=1000)

    width = 5  # base size
    height = 60

    alg = "fqi"
    for a in ["q_visit", "q_count", "eps_greedy", "greedy", "intrinsic", "ucb"]:
        if a in savepath:
            alg = a
            break

    os.makedirs(os.path.join("heatmaps", "n"), exist_ok=True)
    os.makedirs(os.path.join("heatmaps", "q"), exist_ok=True)

    arr = critic.visit_count()
    fig, axs = plt.subplots(1, 1)
    draw(arr[:, 0].squeeze().T, width, height, None, os.path.join("heatmaps", "n", alg + "_off.png"), arr.min(), arr.max(), False)
    draw(arr[:, 1].squeeze().T, width, height, None, os.path.join("heatmaps", "n", alg + "_on.png"), arr.min(), arr.max(), True)
    plt.close()

    arr = critic.q()
    fig, axs = plt.subplots(1, 1)
    draw(arr[:, 0].squeeze().T, width, height, "{x:.2f}", os.path.join("heatmaps", "q", alg + "_off.png"), arr.min(), arr.max(), False)
    draw(arr[:, 1].squeeze().T, width, height, "{x:.2f}", os.path.join("heatmaps", "q", alg + "_on.png"), arr.min(), arr.max(), True)
    plt.close()


if __name__ == "__main__":
    seed = 0
    env = "river_swim"
    mon = "button"
    dir = "heatmaps"
    for alg in ["q_visit", "q_count", "eps_greedy", "greedy", "intrinsic", "ucb"]:
        os.system(f"python main.py environment={env} agent=optimistic_init experiment.debugdir={dir} experiment.testing_episodes=0 monitor={mon} experiment.rng_seed={seed} algorithm={alg}")
    os.system(f"python fqi.py environment={env} experiment.testing_episodes=0 monitor={mon} agent=optimistic_init experiment.debugdir={dir}")
