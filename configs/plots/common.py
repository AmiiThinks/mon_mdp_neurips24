import numpy as np
from collections import defaultdict
import seaborn as sns

smoothing_window = 20
consecutive_steps_for_convergence = 200
y_tick_pad = -20
n_seeds = 100
savedir = "main"

benchmarks = [  # alg, q0_min, q0_max, q0_visit_min, q0_visit_max, eps_init, eps_min, beta_bar, label
    ["q_visit", -10.0, -10.0, 1.0, 1.0, 1.0, 0.0, 0.01, f"Ours"],
    ["greedy", 1.0, 1.0, None, None, 0.0, 0.0, None, f"Optimism"],
    ["ucb", 1.0, 1.0, None, None, 1.0, 0.0, None, f"UCB"],
    ["intrinsic", 1.0, 1.0, None, None, 1.0, 0.0, None, f"Intrinsic"],
    ["eps_greedy", 1.0, 1.0, None, None, 1.0, 0.0, None, f"Naive"],
]

alg_to_label = {
    "q_visit": "Ours",
    "greedy": "Optimism",
    "ucb": "UCB",
    "intrinsic": "Intrinsic",
    "eps_greedy": "Naive",
}

colors = sns.color_palette("colorblind")[:len(alg_to_label)-1]
colors.insert(1, "k")
alg_to_color = {
    k: colors[i] for i, k in enumerate(alg_to_label.keys())
}
