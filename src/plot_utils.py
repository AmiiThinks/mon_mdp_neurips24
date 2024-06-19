from matplotlib.legend_handler import HandlerLine2D
from matplotlib import pyplot as plt
import numpy as np


def make_subplots(nrows, ncols, width_per_plot=2, height_per_plot=2):
    wspace = 0.1
    hspace = 0.1
    width = width_per_plot * (ncols) + wspace * (ncols - 1)
    height = height_per_plot * (nrows) + hspace * (nrows - 1)
    fig, axs = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    return fig, axs


def highlight_cell(ax, x, y, **kwargs):
    rect = plt.Rectangle((x, y), 1, 1, fill=False, **kwargs)
    ax.add_patch(rect)
    return rect


# https://stackoverflow.com/a/42170161/754136
class SymHandler(HandlerLine2D):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        return super(SymHandler, self).create_artists(
            legend, orig_handle, xdescent, 0.6 * height, width, height, fontsize, trans
        )


# https://stackoverflow.com/a/63458548/754136
def smooth(arr, span):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    # re[0] = np.average(arr[:span])
    re[0] = arr[0]
    for i in range(1, span + 1):
        re[i] = np.average(arr[: i + span])
        re[-i] = np.average(arr[-i - span :])
    return re


# def smooth(arr, span):
#     cumsum_vec = np.cumsum(arr)
#     moving_average = (cumsum_vec[2 * span:] - cumsum_vec[:-2 * span]) / (2 * span)
#     front, back = [np.average(arr[:span])], []
#     for i in range(1, span):
#         front.append(np.average(arr[:i + span]))
#         back.insert(0, np.average(arr[-i - span:]))
#     back.insert(0, np.average(arr[-2 * span:]))
#     return np.concatenate((front, moving_average, back))


def error_shade_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    """
    Plot a curve with the data average, and shaded error area for 95% confidence interval.
    You can pass standard pyplot.plot arguments like color, linestyle, linewidth, label, ...
    """

    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)

    (line,) = ax.plot(x, y, **kwargs)

    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])
    ax.fill_between(x, y - error, y + error, alpha=0.2, linewidth=0.0, color=line.get_color())


def error_bar_plot(ax, data, stepsize, smoothing_window=1, **kwargs):
    """
    Plot a curve with the data average, and error bars for 95% confidence interval.
    You can pass standard pyplot.errorbar arguments like color, linestyle, linewidth, label, ...
    """

    y = np.nanmean(data, 0)
    x = np.arange(len(y))
    x = [stepsize * step for step in range(len(y))]
    if smoothing_window > 1:
        y = smooth(y, smoothing_window)

    error = np.nanstd(data, axis=0)
    if smoothing_window > 1:
        error = smooth(error, smoothing_window)
    error = 1.96 * error / np.sqrt(data.shape[0])

    ax.errorbar(x, y, error, **kwargs)
