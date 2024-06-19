import gymnasium
import hydra
from omegaconf import DictConfig
import os
import numpy as np
import git
from tqdm import tqdm

from src.utils import dict_to_id, mesh_combo, set_rng_seed
from src.wrappers import monitor_wrappers
from src.critic import QTableCriticWithVisitQ
from src.actor import EpsilonGreedyWithVisitQ

# This script runs Fitted Q-Iteration to learn the true optimal Q-function.
# (Still approximated because we don't use exact transition probabilities,
# yet quite accurate because we train on 10000 transitions for each (s, a) pair.)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg: DictConfig) -> None:
    set_rng_seed(cfg.experiment.rng_seed)

    group = dict_to_id(cfg.environment) + "/" + dict_to_id(cfg.monitor)
    sha = git.Repo(search_parent_directories=True).head.object.hexsha
    base_folder = os.path.join(sha, group)

    # Fix max Q for infinite horizon MDPs
    if cfg.environment.id in ["RiverSwim"]:
        if cfg.agent.critic.q0_max == 1.0:  # optimistic
            cfg.agent.critic.q0_max = 50.0
        if cfg.agent.critic.q0_min == 1.0:
            cfg.agent.critic.q0_min = 50.0

    env = gymnasium.make(**cfg.environment)
    env = getattr(monitor_wrappers, cfg.monitor.id)(env, **cfg.monitor)
    env.reset(seed=cfg.experiment.rng_seed)

    sizes = (
        env.observation_space["env"].n,
        env.observation_space["mon"].n,
        env.action_space["env"].n,
        env.action_space["mon"].n,
    )
    data = mesh_combo(
        np.arange(sizes[0]),
        np.arange(sizes[1]),
        np.arange(sizes[2]),
        np.arange(sizes[3]),
    )
    transitions_per_pair = 10000
    cfg.agent.critic.lr.init_value = 0.1
    cfg.agent.critic.lr.min_value = 0.01
    cfg.agent.critic.lr.steps = transitions_per_pair * data.shape[0]
    critic = QTableCriticWithVisitQ(*sizes, **cfg.agent.critic)
    actor = EpsilonGreedyWithVisitQ(critic, **cfg.agent.actor)

    def save_pics():
        if cfg.experiment.debugdir is not None:
            from plot_gridworld_agent import plot_agent

            filepath = os.path.join(cfg.experiment.debugdir, base_folder)
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(filepath, "fqi")
            os.makedirs(filepath, exist_ok=True)
            plot_agent(actor, critic, filepath)

    pbar = tqdm(range(transitions_per_pair))
    for i in pbar:
        tot_loss = 0.0
        for d in data:
            obs = {"env": d[0], "mon": d[1]}
            act = {"env": d[2], "mon": d[3]}
            env.set_state(obs)
            next_obs, rwd, term, trunc, _ = env.step(act)
            tot_loss += critic.update(
                np.asarray([obs["env"]]),
                np.asarray([obs["mon"]]),
                np.asarray([act["env"]]),
                np.asarray([act["mon"]]),
                np.asarray([rwd["env"]]),  # fully observable
                np.asarray([rwd["mon"]]),
                np.asarray([term]),
                np.asarray([next_obs["env"]]),
                np.asarray([next_obs["mon"]]),
            ).mean()
        pbar.set_description(f"{tot_loss}")
        if i % 10 == 0:
            save_pics()
        if tot_loss < 1e-4:
            break

    save_pics()


if __name__ == "__main__":
    run()
