from configs.plots.common import *

env_to_label = {
    "iGym-Gridworlds/Empty-Distract-6x6-v0": "Empty 10x10",
    "iGym-Gridworlds/Empty-Loop-3x3-v0": "Loop 3x3",
    "iGym-Gridworlds/Quicksand-Distract-4x4-v0": "Quicksand 4x4",
    "iGym-Gridworlds/TwoRoom-Quicksand-3x5-v0": "Two-Room 3x5",
    "iGym-Gridworlds/TwoRoom-Distract-Middle-2x11-v0": "Two-Room 2x11",
    "iGym-Gridworlds/Corridor-3x4-v0": "Corridor 3x4",
    "iGym-Gridworlds/Straight-20-v0": "Straight 20",
    "iGym-Gridworlds/RiverSwim-6-v0": "River Swim 6",
}

env_to_count = {
    "iGym-Gridworlds/Empty-Distract-6x6-v0": 36 * 5,
    "iGym-Gridworlds/Empty-Loop-3x3-v0": 9 * 5,
    "iGym-Gridworlds/Quicksand-Distract-4x4-v0": 16 * 5,
    "iGym-Gridworlds/TwoRoom-Quicksand-3x5-v0": 15 * 5,
    "iGym-Gridworlds/TwoRoom-Distract-Middle-2x11-v0": 22 * 5,
    "iGym-Gridworlds/Corridor-3x4-v0": 12 * 5,
    "iGym-Gridworlds/Straight-20-v0": 20 * 5,
    "iGym-Gridworlds/RiverSwim-6-v0": 6 * 2,
}

mon_to_label = {
    "iFullMonitor": "No Monitor",
    "iButtonMonitor": "Button",
    "iRandomNonZeroMonitor": "Random Hide (Non-Zero)",
    "iStatelessBinaryMonitor": "Binary",
    "iNMonitor_nm4": "N",
    "iBatteryMonitor_mb10": "Battery",
    "iLevelMonitor_nl3": "Level",
}

mon_to_count = {
    "iFullMonitor": 1,
    "iButtonMonitor": 2,
    "iRandomNonZeroMonitor": 1,
    "iStatelessBinaryMonitor": 2,
    "iNMonitor_nm4": 16,
    "iBatteryMonitor_mb10": 20,
    "iLevelMonitor_nl3": 12,
}
