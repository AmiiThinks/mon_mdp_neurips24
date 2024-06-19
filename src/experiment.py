import gymnasium as gym
import numpy as np
import wandb
from tqdm import tqdm

from src.actor import Actor
from src.critic import Critic
from src.utils import set_rng_seed, cantor_pairing


class Experiment:
    def __init__(
        self,
        env: gym.Env,
        env_test: gym.Env,
        actor: Actor,
        critic: Critic,
        training_steps: int,
        testing_episodes: int,
        testing_points: int,
        rng_seed: int = 1,
        hide_progress_bar: bool = True,
        **kwargs,
    ):
        """
        Args:
            env (gymnasium.Env): environment used to collect training samples,
            env_test (gymnasium.Env): environment used to test the greedy policy,
            actor (Actor): actor to draw actions,
            critic (Critic): critic to evaluate state-action pairs,
            training_steps (int): how many environment steps training will last,
            testing_episodes (int): number of episodes to test the greedy policy,
            testing_points (int): how many testing data points will be saved.
                This and training_steps will determine how often data will be logged.
                For example, if training_steps = 10000 and testing_points = 1000,
                the policy will be tested every 10 training_steps. Note that
                testing_points will always be at least 1 (i.e., testing at step 0)
                and there will also be an extra testing point at the end of training,
            rng_seed (int): to fix random seeds for reproducibility,
            hide_progress_bar (bool): to show tqdm progress bar with some basic info,
        """

        self._env = env
        self._env_test = env_test

        self._actor = actor
        self._critic = critic
        self._gamma = critic.gamma

        self._training_steps = training_steps
        self._testing_episodes = testing_episodes
        self._log_frequency = training_steps // max(testing_points, 1)

        self._rng_seed = rng_seed
        self._hide_progress_bar = hide_progress_bar

    def train(self):
        set_rng_seed(self._rng_seed)
        self._actor.reset()
        self._critic.reset()

        data = {
            "training_steps": self._training_steps,
            "test/return": [],
            "train/return": [],
            "train/loss": [],
            "train/visited_sa": [],
            "train/visited_sa_std": [],
            "train/visited_r": [],
            "train/visited_r_sum": [],
            "train/visited_r_std": [],
            "train/beta": [],
        }

        pbar = tqdm(total=self._training_steps, disable=self._hide_progress_bar)
        tot_steps = 0
        tot_episodes = 0
        last_ep_return = np.nan
        last_ep_loss = np.nan

        def log_and_print():
            # Log greedy policy evaluation
            test_return = self.test()
            test_dict = {"test/return": test_return.mean()}
            wandb.log(test_dict, step=tot_steps, commit=False)
            for k, v in test_dict.items():
                data[k].append(v)

            # Log exploration stats
            visit_count = self._critic.visit_count()
            visited_sa = np.count_nonzero(visit_count)
            visited_sa_std = visit_count.std()
            reward_count = self._critic.reward_count()
            visited_r = np.count_nonzero(reward_count)
            visited_r_sum = reward_count.sum()
            visited_r_std = reward_count.std()
            train_dict = {
                "train/return": last_ep_return,
                "train/loss": last_ep_loss,
                "train/visited_sa": visited_sa,
                "train/visited_sa_std": visited_sa_std,
                "train/visited_r": visited_r,
                "train/visited_r_sum": visited_r_sum,
                "train/visited_r_std": visited_r_std,
                "train/beta": self._actor.beta,
            }
            wandb.log(train_dict, step=tot_steps, commit=False)
            for k, v in train_dict.items():
                data[k].append(v)

            # Upldate progress bar
            pbar.update(tot_steps - pbar.n)
            pbar.set_description(
                f"train[{last_ep_return:.3f}] _ "
                f"test[{np.mean(test_return):.3f}] _ "
                f"visited_sa[{visited_sa}] _ "
                f"visited_sa_std[{visited_sa_std:.3f}] _ "
                f"visited_r[{visited_r}] _ "
                f"visited_r_sum[{visited_r_sum}] _ "
                f"visited_r_std[{visited_r_std:.3f}] _ "
                f"beta[{self._actor.beta:.4f}] _ "
            )  # fmt: skip

        while tot_steps < self._training_steps:
            ep_seed = cantor_pairing(self._rng_seed, tot_episodes)
            ep_generator = np.random.default_rng(seed=ep_seed)
            obs, _ = self._env.reset(seed=ep_seed)
            ep_return = 0.0
            ep_loss = 0.0
            ep_steps = 0
            tot_episodes += 1

            while True:
                if tot_steps % self._log_frequency == 0:
                    log_and_print()

                tot_steps += 1
                act = self._actor(obs["env"], obs["mon"], ep_generator)
                act = {"env": act[0], "mon": act[1]}
                next_obs, rwd, term, trunc, info = self._env.step(act)
                self._critic.update_counts(
                    obs["env"], obs["mon"], act["env"], act["mon"], rwd["proxy"]
                )
                step_loss = self._critic.update(
                    np.asarray([obs["env"]]),
                    np.asarray([obs["mon"]]),
                    np.asarray([act["env"]]),
                    np.asarray([act["mon"]]),
                    np.asarray([rwd["proxy"]]),
                    np.asarray([rwd["mon"]]),
                    np.asarray([term]),
                    np.asarray([next_obs["env"]]),
                    np.asarray([next_obs["mon"]]),
                )
                self._actor.update()

                ep_return += (self._gamma**ep_steps) * (rwd["env"] + rwd["mon"])
                ep_loss += step_loss.mean()
                ep_steps += 1
                obs = next_obs

                if term or trunc or tot_steps >= self._training_steps:
                    break

            last_ep_return = ep_return
            last_ep_loss = ep_loss

        log_and_print()

        self._env.close()
        self._env_test.close()
        pbar.close()

        return data

    def test(self):
        self._actor.eval()
        self._critic.eval()
        ep_return = np.zeros((self._testing_episodes))

        for ep in range(self._testing_episodes):
            ep_seed = cantor_pairing(self._rng_seed, ep)
            ep_generator = np.random.default_rng(seed=ep_seed)
            obs, _ = self._env_test.reset(seed=ep_seed)
            ep_steps = 0
            while True:
                act = self._actor(obs["env"], obs["mon"], ep_generator)
                act = {"env": act[0], "mon": act[1]}
                next_obs, rwd, term, trunc, info = self._env_test.step(act)
                ep_return[ep] += (self._gamma**ep_steps) * (rwd["env"] + rwd["mon"])
                if term or trunc:
                    break
                obs = next_obs
                ep_steps += 1

        self._actor.train()
        self._critic.train()
        return ep_return
