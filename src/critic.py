import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig

import src.parameter as parameter
from src.approximator import MSETable, CountTable, RunningMeanTable


def td_target(rwd: np.array, term: np.array, q_next: np.array, gamma: float):
    """
    Temporal-difference Q-Learning target. Vectorized.

    Args:
        rwd (np.array): r_t,
        term (np.array): True if s_t is terminal, False otherwise,
        q_next (np.array): max_a Q(s_{t+1}, a),
        gammma (float): discount factor,
    """

    return rwd + gamma * (1.0 - term) * q_next


class Critic(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass


class QCritic(Critic):
    """
    Generic class for Mon-MDP critics.
    """

    def __init__(self, gamma: float, lr: DictConfig, **kwargs):
        """
        Args:
            gamma (float): discount factor,
            lr (DictConfig): configuration to initialize the learning rate,
        """

        self.gamma = gamma
        self.lr = getattr(parameter, lr.id)(**lr)
        self.q = None  # Q-function
        self.q_target = None  # for computing Q(next_state) in the TD target
        self.r = None  # reward model

    def reset(self):
        self.q.reset()
        self.q_target.reset()
        self.r.reset()

    def update(
        self,
        obs_env, obs_mon,
        act_env, act_mon,
        rwd_proxy, rwd_mon,
        term,
        next_obs_env, next_obs_mon,
    ):  # fmt: skip
        self.update_r(obs_env, act_env, rwd_proxy)
        rwd = self.r(obs_env, act_env) + rwd_mon
        q_next = self.q_target(next_obs_env, next_obs_mon)
        target = td_target(rwd, term, q_next.max((-2, -1)), self.gamma)
        error = self.q.update(
            obs_env, obs_mon,
            act_env, act_mon,
            target=target,
            stepsize=self.lr.value,
        )  # fmt: skip
        self.lr.step()
        return error

    def update_r(self, obs_env, act_env, rwd_proxy):
        i = np.logical_not(np.isnan(rwd_proxy))
        if np.any(i):
            self.r.update(obs_env[i], act_env[i], target=rwd_proxy[i])


class QTableCritic(QCritic):
    """
    Instance of QCritic that uses tabular Q-functions.
    It also keeps counts that are updated by the data collection procedure.
    """

    def __init__(
        self,
        n_obs_env: int, n_obs_mon: int,
        n_act_env: int, n_act_mon: int,
        q0_min: float, q0_max: float,
        r0_min: float, r0_max: float,
        **kwargs
    ):  # fmt: skip
        QCritic.__init__(self, **kwargs)
        self.act_shape = (n_act_env, n_act_mon)
        self.obs_shape = (n_obs_env, n_obs_mon)
        self.env_shape = (n_obs_env, n_act_env)
        self.mon_shape = (n_obs_mon, n_act_mon)
        self.q = MSETable(
            *self.obs_shape,
            *self.act_shape,
            init_value_min=q0_min,
            init_value_max=q0_max,
        )
        self.q_target = self.q  # with tabular Q we don't need a different target
        self.r = RunningMeanTable(
            *self.env_shape,
            init_value_min=r0_min,
            init_value_max=r0_max,
        )
        self.reward_count = CountTable(*self.env_shape)  # N(sE,aE) when reward is observed
        self.visit_count = CountTable(*self.obs_shape, *self.act_shape)  # N(s,a)
        QCritic.reset(self)

    def update_counts(self, obs_env, obs_mon, act_env, act_mon, rwd_proxy):
        i = np.logical_not(np.isnan(rwd_proxy))
        if np.any(i):
            self.reward_count.update(obs_env[i], act_env[i])
        self.visit_count.update(obs_env, obs_mon, act_env, act_mon)


class QTableCriticWithVisitReward(QTableCritic):
    """
    Intrinsic reward based on the inverse of the visitation count.
    """

    def update(
        self,
        obs_env, obs_mon,
        act_env, act_mon,
        rwd_proxy, rwd_mon,
        term,
        next_obs_env, next_obs_mon,
    ):  # fmt: skip
        n = self.visit_count(obs_env, obs_mon, act_env, act_mon)
        rwd_intrinsic = 1.0 / np.sqrt(n)
        rwd_coeff = 1.0 - self.gamma_visit
        return QCritic.update(self,
            obs_env, obs_mon,
            act_env, act_mon,
            rwd_proxy, rwd_mon + rwd_coeff * rwd_intrinsic,
            term,
            next_obs_env, next_obs_mon,
        )  # fmt: skip


class QTableCriticWithVisitQ(QTableCritic):
    """
    This critic learns S-functions (Q-functions based on the successor representation).
    In the code, they are called Q-visit.
    """

    def __init__(
        self,
        n_obs_env: int, n_obs_mon: int,
        n_act_env: int, n_act_mon: int,
        q0_visit_min: float, q0_visit_max: float,
        gamma_visit: float,
        lr_visit: DictConfig,
        **kwargs,
    ):  # fmt: skip
        QTableCritic.__init__(self, n_obs_env, n_obs_mon, n_act_env, n_act_mon, **kwargs)
        self.lr_visit = getattr(parameter, lr_visit.id)(**lr_visit)
        self.gamma_visit = gamma_visit
        self.q_visit = MSETable(
            *self.obs_shape,
            *self.act_shape,
            n_obs_env * n_obs_mon * n_act_env * n_act_mon,
            init_value_min=q0_visit_min,
            init_value_max=q0_visit_max,
        )
        self.q_visit_target = self.q_visit
        QTableCriticWithVisitQ.reset(self)

    def reset(self):
        self.q_visit.reset()
        self.q_visit_target.reset()
        QCritic.reset(self)

    def update(
        self,
        obs_env, obs_mon,
        act_env, act_mon,
        rwd_proxy, rwd_mon,
        term,
        next_obs_env, next_obs_mon,
    ):  # fmt: skip
        q_visit_next = self.q_visit_target(next_obs_env, next_obs_mon)
        rwd_visit = np.zeros((q_visit_next.shape[0], q_visit_next.shape[-1]))
        idx = np.ravel_multi_index((obs_env, obs_mon, act_env, act_mon), self.q.shape)
        rwd_visit[:, idx] = 1.0
        target = td_target(
            rwd_visit,
            np.logical_or(term[:, None], rwd_visit),
            q_visit_next.max((-3, -2)),
            self.gamma_visit,
        )
        error = self.q_visit.update(
            obs_env, obs_mon,
            act_env, act_mon,
            target=target,
            stepsize=self.lr_visit.value,
        )  # fmt: skip
        self.lr_visit.step()

        return error.mean(-1) + QCritic.update(self,
            obs_env, obs_mon,
            act_env, act_mon,
            rwd_proxy, rwd_mon,
            term,
            next_obs_env, next_obs_mon,
        )  # fmt: skip
