import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


# https://stackoverflow.com/a/42170161/754136
from matplotlib.legend_handler import HandlerLine2D


class SymHandler(HandlerLine2D):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        return super(SymHandler, self).create_artists(
            legend, orig_handle, xdescent, 0.6 * height, width, height, fontsize, trans
        )


sns.set_context("paper")
# sns.set_style('whitegrid', {'legend.frameon': True})
sns.set_style("darkgrid", {"legend.frameon": True})
plt.rcParams["axes.axisbelow"] = False
# plt.rcParams["axes.grid"] = False
plt.rcParams["grid.linestyle"] = "--"
# plt.rcParams['font.family'] = 'DejaVu Sans Mono'
plt.rcParams["font.family"] = "Bree Serif"
font_size = 12
plt.rcParams["font.size"] = font_size


labels = [
    "Ours",
    "Optimism",
    "UCB",
    "Intrinsic Reward",
    "Naive",
]
colors = sns.color_palette("colorblind")[: len(labels) - 1]
colors.insert(1, "k")

fig, axs = plt.subplots(2, 1)
figl = plt.figure()
legend_handles = []
legend_labels = []
for lab, c in zip(labels, colors):
    axs[0].plot(1, 1, label=lab, lw=3, c=c)
    handles, labels = axs[0].get_legend_handles_labels()
    legend_handles.extend(handles)
    legend_labels.extend(labels)

# Remove duplicate legends
unique_legend = [
    (h, l)
    for i, (h, l) in enumerate(zip(legend_handles, legend_labels))  # noqa: E741
    if l not in legend_labels[:i]
]

leg = figl.legend(
    *zip(*unique_legend),
    handler_map={matplotlib.lines.Line2D: SymHandler()},
    handleheight=2.4,
    labelspacing=0.05,
    handletextpad=0.6,
    prop={"size": font_size - 4},
    # loc='upper left', bbox_to_anchor=(1, 1.5), # right, outside
    loc="center",
    bbox_to_anchor=(0.485, -0.5),  # below, outside
    ncol=len(labels),
    columnspacing=1.2,
)


legend_handles = []
legend_labels = []
axs[1].plot(1, 1, label="Straight Line: Optimistic Initialization", lw=3, ls="-", c="k", alpha=0.5)
axs[1].plot(1, 1, label="Dashed Line: Pessimistic Initialization", lw=3, ls="--", c="k", alpha=0.5)
handles, labels = axs[1].get_legend_handles_labels()
legend_handles.extend(handles)
legend_labels.extend(labels)

leg = figl.legend(
    handles,
    labels,
    handler_map={matplotlib.lines.Line2D: SymHandler()},
    handleheight=2.4,
    labelspacing=0.05,
    handletextpad=0.6,
    handlelength=0,
    prop={"size": font_size - 4},
    # loc='upper left', bbox_to_anchor=(1, 1.5), # right, outside
    loc="center",
    bbox_to_anchor=(0.485, -0.56),  # below, outside
    ncol=len(labels),
    columnspacing=1.2,
)

plt.draw()

plt.savefig("legend.png", bbox_inches="tight", pad_inches=0, dpi=1500)
