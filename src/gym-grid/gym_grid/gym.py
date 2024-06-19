from gymnasium.envs.registration import register


def register_envs():
    register(
        id="Gridworld-Straight-20-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=200,
        kwargs={
            "grid": "20_straight",
        },
    )

    register(
        id="Gridworld-Empty-2x2-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=10,
        kwargs={
            "grid": "2x2_empty",
        },
    )

    register(
        id="Gridworld-Empty-3x3-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_empty",
        },
    )

    register(
        id="Gridworld-Empty-Loop-3x3-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_empty_loop",
        },
    )

    register(
        id="Gridworld-Empty-10x10-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=100,
        kwargs={
            "grid": "10x10_empty",
        },
    )

    register(
        id="Gridworld-Empty-Distract-6x6-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "6x6_distract",
        },
    )

    register(
        id="Gridworld-Penalty-3x3-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_penalty",
        },
    )

    register(
        id="Gridworld-Quicksand-4x4-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "4x4_quicksand",
        },
    )

    register(
        id="Gridworld-Quicksand-Distract-4x4-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "4x4_quicksand_distract",
        },
    )

    register(
        id="Gridworld-TwoRoom-Quicksand-3x5-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x5_two_room_quicksand",
        },
    )

    register(
        id="Gridworld-Corridor-3x4-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x4_corridor",
        },
    )
    register(
        id="Gridworld-Full-5x5-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "5x5_full",
        },
    )

    register(
        id="Gridworld-TwoRoom-Distract-Middle-2x11-v0",
        entry_point="gym_grid.gridworld:GridworldMiddleStart",
        max_episode_steps=200,
        kwargs={
            "grid": "2x11_two_room_distract",
        },
    )

    register(
        id="Gridworld-Barrier-5x5-v0",
        entry_point="gym_grid.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "5x5_barrier",
        },
    )

    register(
        id="RiverSwim-6-v0",
        entry_point="gym_grid.gridworld:RiverSwim",
        max_episode_steps=200,
        kwargs={
            "grid": "river_swim_6",
        },
    )
