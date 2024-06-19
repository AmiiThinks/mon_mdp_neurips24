import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.utils import random_argmax
import src.parameter as parameter
from src.critic import Critic
from src.approximator import CountTable


def greedy(x, rng_generator):
    return tuple(random_argmax(x, rng_generator))


def softmax(x: np.array, eps: float, rng_generator):
    x_exp = np.exp((x - x.max()) / max(eps, 1e-12))
    p = x_exp / x_exp.sum()
    shp = p.shape
    p = p.flatten()
    y = rng_generator.choice(len(p), p=p)
    return np.unravel_index(y, shp)


class Actor(ABC):
    def __init__(self, critic: Critic):
        self._critic = critic
        self._train = True
        self.beta = np.nan
        self.visit_goal_count = CountTable(*self._critic.visit_count.shape)
        self.reset()

    def __call__(self, obs_env, obs_mon, rng_generator=None):
        """
        Draw one action in one state. Not vectorized.
        """

        if self._train:
            return self.explore(obs_env, obs_mon, rng_generator)
        else:
            return greedy(self._critic.q(obs_env, obs_mon), rng_generator)

    @abstractmethod
    def explore(self, obs_env, obs_mon, rng_generator):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def eval(self):
        self._train = False

    def train(self):
        self._train = True


class EpsilonGreedy(Actor):
    def __init__(self, critic: Critic, eps: DictConfig, **kwargs):
        """
        Args:
            critic (Critic): the critic providing estimates of state-action values,
            eps (DictConfig): configuration to initialize the exploration coefficient
                epsilon,
        """

        self._eps = getattr(parameter, eps.id)(**eps)
        Actor.__init__(self, critic)

    def explore(self, obs_env, obs_mon, rng_generator):
        if rng_generator.random() < self._eps.value:
            return tuple(rng_generator.integers(self._critic.act_shape))
        else:
            return greedy(self._critic.q(obs_env, obs_mon), rng_generator)

    def update(self):
        self._eps.step()

    def reset(self):
        self._eps.reset()


class EpsilonGreedyBalancedWandering(EpsilonGreedy):
    """
    Inspired by "Near-Optimal Reinforcement Learning in Polynomial Time".
    Like epsilon-greedy, but instead of a random action the agent selects the
    action with the lowest count.
    """

    def explore(self, obs_env, obs_mon, rng_generator):
        if rng_generator.random() < self._eps.value:
            return greedy(-self._critic.visit_count()[obs_env, obs_mon])
        else:
            return greedy(self._critic.q(obs_env, obs_mon))


class EpsilonGreedyWithUCB(EpsilonGreedy):
    """
    Actor with UCB exploration.
    """

    def explore(self, obs_env, obs_mon, rng_generator):
        if rng_generator.random() < self._eps.value:
            return tuple(rng_generator.integers(self._critic.act_shape))
        else:
            n = self._critic.visit_count(obs_env, obs_mon)
            ucb = np.sqrt(2.0 * np.log(n.sum()) / n)
            ucb[np.isnan(ucb)] = np.inf
            return greedy(self._critic.q(obs_env, obs_mon) + ucb, rng_generator)


class EpsilonGreedyWithVisitQ(EpsilonGreedy):
    """
    Actor for Q-visit critics.
    """

    def __init__(self, critic: Critic, beta_bar: float, **kwargs):
        """
        Args:
            critic (Critic): the critic providing estimates of state-action values,
            beta_bar (float): ratio threshold for exploration / exploitation,
        """

        EpsilonGreedy.__init__(self, critic, **kwargs)
        self._beta_bar = beta_bar
        self.t = 1.0
        self.beta = np.inf

    def explore(self, obs_env, obs_mon, rng_generator):
        self.t += 1

        n = self._critic.visit_count()
        goal = n.argmin()
        goal = np.unravel_index(goal, n.shape)
        beta = np.log(self.t) / n[goal]
        self.beta = beta  # the experiment reads it and logs it

        if beta > self._beta_bar:
            self.visit_goal_count.update(*goal)
            if rng_generator.random() < self._eps.value:
                return tuple(rng_generator.integers(self._critic.act_shape))
            goal = np.ravel_multi_index(goal, self._critic.q.shape)
            return greedy(self._critic.q_visit(obs_env, obs_mon)[..., goal], rng_generator)
        else:
            return greedy(self._critic.q(obs_env, obs_mon), rng_generator)
