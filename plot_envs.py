import gymnasium as gym
import matplotlib.pyplot as plt

id = [
    "Gym-Grid/Gridworld-Corridor-3x4-v0",
    "Gym-Grid/Gridworld-Empty-10x10-v0",
    "Gym-Grid/Gridworld-Quicksand-Distract-4x4-v0",
    "Gym-Grid/Gridworld-Empty-Loop-3x3-v0",
    "Gym-Grid/Gridworld-Straight-20-v0",
    "Gym-Grid/Gridworld-TwoRoom-3x5-v0",
    "Gym-Grid/RiverSwim-10-v0",
    "Gym-Grid/RiverSwim-6-v0",
]

for i in id:
    env = gym.make(i, render_mode="rgb_array")
    env.reset()
    img = env.render()
    plt.axis("off")
    plt.imshow(img)
    plt.draw()
    plt.savefig(i.replace("/", "_") + "png", bbox_inches="tight", pad_inches=0, dpi=1500)
    env.close()
