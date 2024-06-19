# fmt: off
from setuptools import setup

packages = ["gym_grid"]
install_requires = [
    "gymnasium",
    "pygame"
]
entry_points = {
    "gymnasium.envs": ["Gym-Grid=gym_grid.gym:register_envs"]
}

setup(
    name="Gym-Grid",
    version="0.0.1",
    license="GPL",
    packages=packages,
    entry_points=entry_points,
    install_requires=install_requires,
)
