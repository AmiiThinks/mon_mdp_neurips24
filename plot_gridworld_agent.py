# ruff: noqa: F403, F405
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import os
from src.plot_utils import highlight_cell

plt.rcParams["font.family"] = "Bree Serif"
plt.rcParams["font.size"] = 14


def plot_agent(actor, critic, savepath=""):
    """
    Plot and save the Q-table, counts, and other info.
    """

    def reshape(arr):
        a, b, c, d = arr.shape
        arr = arr.reshape(a * b, c * d, order="F")
        arr = np.insert(arr, np.arange(a, a * b, a), np.nan, 0)
        arr = np.insert(arr, np.arange(c, c * d, c), np.nan, 1)
        return arr

    def draw(
        arr,
        width,
        height,
        xticks,
        xlabels,
        yticks,
        ylabels,
        format,
        filepath,
        colorbar=True,
        highlight_max=False,
    ):
        fig, axs = plt.subplots(1, 1)
        fig.set_figwidth(width)
        fig.set_figheight(height)
        im = axs.imshow(arr)
        axs.set_xticks(xticks, xlabels)
        axs.set_yticks(yticks, ylabels)
        for (j, i), label in np.ndenumerate(arr):
            if not np.isnan(label):
                axs.text(i, j, format(label), ha="center", va="center")
                if highlight_max:
                    if arr[j, i] == np.nanmax(arr[j, :]):
                        highlight_cell(axs, i - 0.5, j - 0.5, color=(0.9, 0, 0), linewidth=1.5)
                    if arr[j, i] == np.nanmax(arr):
                        highlight_cell(axs, i - 0.5, j - 0.5, color=(1.0, 0, 0), linewidth=3.0)
        if colorbar:
            divider = make_axes_locatable(axs)
            cax = divider.append_axes("right", size="5%", pad=0.25)
            plt.colorbar(im, cax=cax, format=format, ticks=[np.nanmin(arr), np.nanmax(arr)])
        plt.draw()
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=100)
        plt.close(fig)

    width = 5  # base size
    height = 60

    # Get sizes for ticks
    obs_e, obs_m, act_e, act_m = critic.q.shape
    yticks_e = range(obs_e)
    yticks_em = range((obs_e + 1) * obs_m)[:-1]
    ylabels_e = list(range(obs_e))
    ylabels_em = ((list(range(obs_e)) + [""]) * obs_m)[: len(yticks_em)]
    xticks_e = range(act_e)
    xticks_em = range((act_e + 1) * act_m)[:-1]
    xlabels_e = list(range(act_e))
    xlabels_em = ((list(range(act_e)) + [""]) * act_m)[: len(xticks_em)]
    int_fmt = FormatStrFormatter("%.0f")
    float_fmt = FormatStrFormatter("%.3f")

    args_mdp = {
        "width": width,
        "height": height,
        "xticks": xticks_e,
        "xlabels": xlabels_e,
        "yticks": yticks_e,
        "ylabels": ylabels_e,
    }
    args_mon_mdp = {
        "width": width * act_m,
        "height": height * obs_m,
        "xticks": xticks_em,
        "xlabels": xlabels_em,
        "yticks": yticks_em,
        "ylabels": ylabels_em,
    }

    # Q
    draw(
        reshape(critic.q()),
        format=float_fmt,
        filepath=os.path.join(savepath, "q_function.png"),
        highlight_max=True,
        **args_mon_mdp,
    )

    # R
    draw(
        critic.r(),
        format=float_fmt,
        filepath=os.path.join(savepath, "reward_model.png"),
        **args_mdp,
    )

    # N_r
    draw(
        critic.reward_count(),
        format=int_fmt,
        filepath=os.path.join(savepath, "counts_reward.png"),
        **args_mdp,
    )

    # N_sa
    draw(
        reshape(critic.visit_count()),
        format=int_fmt,
        filepath=os.path.join(savepath, "counts_visit.png"),
        **args_mon_mdp,
    )

    # N_sGaG
    draw(
        reshape(actor.visit_goal_count()),
        format=int_fmt,
        filepath=os.path.join(savepath, "counts_visit_goal.png"),
        **args_mon_mdp,
    )

    # Q_visit
    try:
        q_sa = critic.q_visit()
        for idx in range(q_sa.shape[-1]):
            sa = np.unravel_index(idx, q_sa.shape[:-1])
            draw(
                reshape(q_sa[..., idx]),
                format=float_fmt,
                filepath=os.path.join(savepath, f"q_visit_{sa}.png"),
                highlight_max=True,
                **args_mon_mdp,
            )
    except:
        pass
