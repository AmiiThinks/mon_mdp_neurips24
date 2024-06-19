import gymnasium
from gymnasium import spaces
import numpy as np
from abc import abstractmethod
import pygame


class Monitor(gymnasium.Wrapper):
    """
    Generic class for monitors that DO NOT depend on the environment state.
    Monitors that DO depend on the environment state need to be customized
    according to the environment.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        observability (float): probability that the monitor works properly.
            If < 1.0, then there is a chance that the environment reward is
            unobservable regardless of the state and action.
    """

    def __init__(self, env, observability=1.0, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.observability = observability

    @abstractmethod
    def _monitor_step(self, action, env_reward):
        pass

    @abstractmethod
    def _monitor_set_state(self, state):
        pass

    @abstractmethod
    def _monitor_get_state(self):
        pass

    def set_state(self, state):
        self.env.unwrapped.set_state(state["env"])
        self._monitor_set_state(state["mon"])

    def get_state(self):
        return {"env": self.env.unwrapped.get_state(), "mon": self._monitor_get_state()}

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = self.observation_space["mon"].sample()
        return {"env": env_obs, "mon": self.monitor_state}, env_info

    def render(self):
        """
        Make the screen flash if the agent observes a reward.
        Works only if the environment is rendered with pygame.
        """

        surf = pygame.display.get_surface()
        if surf is not None:
            surf.fill((255, 255, 255, 128), rect=None, special_flags=0)
            pygame.display.update()

    def step(self, action):
        """
        This type of monitors DO NOT depend on the environment state.
        Therefore, we first execute self.env.step() and then self._monitor_step().
        Everything else works as in classic Gymnasium environments, but state,
        actions, and rewards are dictionaries. That is, the agent expects

            actions = {"env": action_env, "mon": action_mon}

        and returns

            state = {"env": state_env, "mon": state_mon}
            reward = {"env": reward_env, "mon": reward_mon, "proxy": reward_proxy}
            terminated = env_terminated or monitor_terminated

        Truncated and info remain the same as self.env.step().
        """
        (
            env_obs,
            env_reward,
            env_terminated,
            env_truncated,
            env_info,
        ) = self.env.step(action["env"])

        (
            monitor_obs,
            proxy_reward,
            monitor_reward,
            monitor_terminated,
        ) = self._monitor_step(action, env_reward)

        obs = {"env": env_obs, "mon": monitor_obs}
        reward = {"env": env_reward, "mon": monitor_reward, "proxy": proxy_reward}
        terminated = env_terminated or monitor_terminated
        truncated = env_truncated

        if self.observability < 1.0 and self.np_random.random() > self.observability:
            reward["proxy"] = np.nan

        if self.render_mode == "human" and not np.isnan(reward["proxy"]):
            self.render()

        return obs, reward, terminated, truncated, env_info


class FullMonitor(Monitor):
    """
    This monitor always shows the true reward, regardless of its state and action.
    The monitor reward is always 0.
    This is a 'trivial Mon-MDP', i.e., it is equivalent to a classic MDP.

    Args:
        env (gymnasium.Env): the Gymnasium environment.
    """

    def __init__(self, env, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip

    def _monitor_set_state(self, state):
        return

    def _monitor_get_state(self):
        return np.array(0)

    def _monitor_step(self, action, env_reward):
        return self._monitor_get_state(), env_reward, 0.0, False


class RandomNonZeroMonitor(Monitor):
    """
    This monitor randomly makes non-zero rewards unobservable.
    There are no monitor states and actions.
    The monitor reward is always 0.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        prob (float): the probability that the reward is unobservable.
    """

    def __init__(self, env, prob=0.5, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.prob = prob

    def _monitor_set_state(self, state):
        return

    def _monitor_get_state(self):
        return np.array(0)

    def _monitor_step(self, action, env_reward):
        monitor_reward = 0.0
        if env_reward != 0 and self.np_random.random() < self.prob:
            proxy_reward = np.nan
        else:
            proxy_reward = env_reward
        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class RandomMonitor(Monitor):
    """
    This monitor randomly makes rewards unobservable.
    Each reward has a different probability of being observed, which is fixed
    when the environment is created.
    There are no monitor states and actions.
    The monitor reward is always 0.

    Args:
        env (gymnasium.Env): the Gymnasium environment.
    """

    def __init__(self, env, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.prob = env.np_random.random((env.observation_space.n, env.action_space.n))

    def _monitor_set_state(self, state):
        return

    def _monitor_get_state(self):
        return np.array(0)

    def _monitor_step(self, action, env_reward):
        monitor_reward = 0.0
        if self.np_random.random() < self.prob[self.unwrapped.get_state(), action["env"]]:
            proxy_reward = np.nan
        else:
            proxy_reward = env_reward
        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class StatefulBinaryMonitor(Monitor):
    """
    Simple monitor where the action is "turn on monitor" / "do nothing".
    The monitor state is also binary ("monitor on" / "monitor off").
    The monitor reward is a constant penalty given if the monitor is on or turned on.

    The monitor can turn off itself randomly at every time step (default probability is 0).
    If the monitor is on, the environment reward is observed.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_cost (float): cost for monitor being active,
        monitor_reset_prob (float): probability of the monitor resetting itself.
    """

    def __init__(self, env, monitor_cost=0.2, monitor_reset_prob=0.0, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.monitor_state = 0  # off
        self.monitor_reset_prob = monitor_reset_prob
        self.monitor_cost = monitor_cost

    def _monitor_set_state(self, state):
        self.monitor_state = state

    def _monitor_get_state(self):
        return np.array(self.monitor_state)

    def _monitor_step(self, action, env_reward):
        if self.monitor_state == 1:
            proxy_reward = env_reward
            monitor_reward = -self.monitor_cost
        else:
            proxy_reward = np.nan
            monitor_reward = 0.0

        if action["mon"] == 1:
            self.monitor_state = 1
        elif action["mon"] == 0:
            pass
        else:
            raise ValueError("illegal monitor action")

        if self.np_random.random() < self.monitor_reset_prob:
            self.monitor_state = 0

        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class StatelessBinaryMonitor(Monitor):
    """
    Simple monitor where the action is "turn on monitor" / "do nothing".
    The monitor is always off. The reward is seen only when the agent asks for it.
    The monitor reward is a constant penalty given if the agent asks to see the reward.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_cost (float): cost for asking the monitor for rewards.
    """

    def __init__(self, env, monitor_cost=0.2, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.monitor_cost = monitor_cost

    def _monitor_set_state(self, state):
        return

    def _monitor_get_state(self):
        return np.array(0)

    def _monitor_step(self, action, env_reward):
        if action["mon"] == 1:
            proxy_reward = env_reward
            monitor_reward = -self.monitor_cost
        else:
            proxy_reward = np.nan
            monitor_reward = 0.0
        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class NMonitor(Monitor):
    """
    There are N monitors. At every time step, a random monitor is on.
    If the agent's action matches the monitor state, the agent observes the
    environment reward but receives a negative monitor reward.
    Otherwise it does not observe the environment reward, but receives a smaller
    positive monitor reward.
    For example, if state = 2 and action = 2, the agent observes the environment
    reward and gets reward_monitor = -0.2.
    If state = 2 and action != 2, the agent does not observe the reward but
    gets reward_monitor = 0.001.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        n_monitors (int): number of monitors,
        monitor_cost (float): cost for observing the reward,
        monitor_bonus (float): reward for not observing the reward.
    """

    def __init__(self, env, n_monitors=5, monitor_cost=0.2, monitor_bonus=0.001, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(n_monitors),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(n_monitors),
        })  # fmt: skip
        self.monitor_state = 0
        self.monitor_cost = monitor_cost
        self.monitor_bonus = monitor_bonus

    def _monitor_set_state(self, state):
        self.monitor_state = state

    def _monitor_get_state(self):
        return np.array(self.monitor_state)

    def _monitor_step(self, action, env_reward):
        assert (
            action["mon"] < self.action_space["mon"].n
        ), "illegal monitor action"  # fmt: skip

        if action["mon"] == self.monitor_state:
            proxy_reward = env_reward
            monitor_reward = -self.monitor_cost
        else:
            proxy_reward = np.nan
            monitor_reward = self.monitor_bonus

        self.monitor_state = self.observation_space["mon"].sample()

        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class LevelMonitor(Monitor):
    """
    The monitor has N levels, from 0 to N - 1.
    The initial level is random, and it increases if the agent's action matches
    the current level.
    For example, if state = 2 and action = 2, then next_state = 3.
    If the agent executes the wrong action, the level resets to 0.
    Actions 0 to N - 1 are costly.
    Action N does nothing and costs nothing.
    Environment rewards will become visible only when the monitor level is max,
    i.e., when state = N - 1.
    To keep it maxxed, the agent has to keep doing action = N - 1 (paying a cost)
    or do action = N (no cost).

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        n_levels (int): number of levels,
        monitor_cost (float): cost for leveling up the monitor state.
    """

    def __init__(self, env, n_levels=4, monitor_cost=0.2, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(n_levels + 1),  # last action is "do nothing"
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(n_levels),
        })  # fmt: skip
        self.monitor_state = 0
        self.monitor_cost = monitor_cost

    def _monitor_set_state(self, state):
        self.monitor_state = state

    def _monitor_get_state(self):
        return np.array(self.monitor_state)

    def _monitor_step(self, action, env_reward):
        assert (
            action["mon"] < self.action_space["mon"].n
        ), "illegal monitor action"  # fmt: skip

        monitor_reward = 0.0
        proxy_reward = np.nan

        if self.monitor_state == self.observation_space["mon"].n - 1:
            proxy_reward = env_reward

        if action["mon"] == self.action_space["mon"].n - 1:
            pass  # last action is "do nothing"
        else:
            monitor_reward = -self.monitor_cost  # pay cost
            if action["mon"] == self.monitor_state:
                self.monitor_state += 1  # raise level
                if self.monitor_state > self.observation_space["mon"].n - 1:  # level is already max
                    self.monitor_state = self.observation_space["mon"].n - 1
            else:
                self.monitor_state = 0  # reset level

        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class LimitedTimeMonitor(Monitor):
    """
    The monitor is on at the beginning of the episode and the agent sees
    rewards for free.
    At every step, there is a small probability that the monitor goes off.
    If it goes off, it stays off until the end of the episode.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_reset_prob (float): probability of the monitor resetting itself.
    """

    def __init__(self, env, monitor_reset_prob=0.2, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.monitor_state = 1
        self.monitor_reset_prob = monitor_reset_prob

    def _monitor_set_state(self, state):
        self.monitor_state = state

    def _monitor_get_state(self):
        return np.array(self.monitor_state)

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = 1  # monitor starts on
        return {"env": env_obs, "mon": self._monitor_get_state()}, env_info

    def _monitor_step(self, action, env_reward):
        monitor_reward = 0.0
        if self.monitor_state == 1:
            proxy_reward = env_reward
        else:
            proxy_reward = np.nan

        if self.np_random.random() < self.monitor_reset_prob:
            self.monitor_state = 0

        return self._monitor_get_state(), proxy_reward, monitor_reward, False


class BatteryMonitor(Monitor):
    """
    The monitor has a battery that is consumed whenever it is on.
    The state of the monitor is the battery level.
    The battery level goes from 0, 1, 2, ..., N.
    Every time the agent asks for monitoring, the battery goes down by 1 level.
    When the battery level reaches 0 the episode ends (the state is terminal).

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        max_battery (int): how many times the agent can ask for monitoring.
    """

    def __init__(self, env, max_battery=5, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(2),  # aks or not
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(max_battery + 1),
        })  # fmt: skip
        self.max_battery = max_battery
        self.monitor_battery = max_battery

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_battery = self.max_battery  # battery full
        return {"env": env_obs, "mon": self._monitor_get_state()}, env_info

    def _monitor_set_state(self, state):
        self.monitor_battery = state

    def _monitor_get_state(self):
        return np.array(self.monitor_battery)

    def _monitor_step(self, action, env_reward):
        proxy_reward = np.nan
        monitor_reward = 0.0
        monitor_terminated = self.monitor_battery == 0

        if action["mon"] == 1:  # ask
            if self.monitor_battery > 0:
                proxy_reward = env_reward
                self.monitor_battery = max(self.monitor_battery - 1, 0)
        elif action["mon"] == 0:  # do nothing
            pass
        else:
            raise ValueError("illegal monitor action")

        return self._monitor_get_state(), proxy_reward, monitor_reward, monitor_terminated


class ButtonMonitor(Monitor):
    """
    Monitor for Gridworlds.
    The monitor is turned on/off by doing LEFT (environment action) where a button is.
    If the monitor is on, the agent receives negative monitor rewards and observes
    the environment rewards.
    Ending an episode with the monitor on results in a large penalty.
    The monitor on/off state at the beginning of an episode is random.
    The position of the button can be specified by an argument (top-left by default).

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_cost (float): cost for monitor being active,
        monitor_end_cost (float): cost for ending an episode (by termination,
            not truncation) with the monitor active,
        button_cell_id (int): position of the monitor,
        env_action_push (int): the environment action to turn the monitor on/off.
    """

    def __init__(self, env, monitor_cost=0.2, monitor_end_cost=2.0, button_cell_id=0, **kwargs):
        Monitor.__init__(self, env, **kwargs)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),  # no monitor action
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(2),  # monitor on/off
        })  # fmt: skip
        self.button_cell_id = button_cell_id
        self.monitor_state = 0
        self.monitor_cost = monitor_cost
        self.monitor_end_cost = monitor_end_cost

    def _monitor_set_state(self, state):
        self.monitor_state = state

    def _monitor_get_state(self):
        return np.array(self.monitor_state)

    def step(self, action):
        env_obs = self.env.unwrapped.get_state()
        (
            env_next_obs,
            env_reward,
            env_terminated,
            env_truncated,
            env_info,
        ) = self.env.step(action["env"])

        monitor_reward = 0.0
        proxy_reward = np.nan
        if self.monitor_state == 1:
            proxy_reward = env_reward
            monitor_reward += -self.monitor_cost
            if env_terminated:
                monitor_reward += -self.monitor_end_cost
        if action["env"] == 0 and env_obs == self.button_cell_id:  # 0 is LEFT
            if self.monitor_state == 1:
                self.monitor_state = 0
            elif self.monitor_state == 0:
                self.monitor_state = 1
        monitor_terminated = False

        obs = {"env": env_next_obs, "mon": self._monitor_get_state()}
        reward = {"env": env_reward, "mon": monitor_reward, "proxy": proxy_reward}
        terminated = env_terminated or monitor_terminated
        truncated = env_truncated

        if self.render_mode == "human" and not np.isnan(reward["proxy"]):
            self.render()

        return obs, reward, terminated, truncated, env_info
