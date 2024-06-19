import numpy as np
from abc import ABC, abstractmethod


class FunctionApproximator(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def __call__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def copy_from(self):
        pass


class Table(FunctionApproximator):
    """
    Multi-dimensional np.array for discrete inputs.
    When called, it expects inputs (such as states and actions) passed as either
    integers, list of integers, or np.array.
    For example, the following code initializes a Q-function for an MDP with
    9 states and 4 actions, and then asks for the value of three state-action pairs.

        >>> q_fun = Table(9, 4)
        >>> q_fun.shape
        (9, 4)
        >>> state = np.array([0, 1, 2])
        >>> value = q_fun(state)
        >>> value.shape
        (3, 4)
        >>> action = np.array([2, 3, 0])
        >>> value = q_fun(state, action)
        >>> value.shape
        (3,)

    This class also works with multi-dimensional states and actions. In the
    following example, both the state and the action are 2-dimensional, with 9x2
    and 4x2 states and actions, respectively.

        >>> q_fun = Table(9, 2, 4, 2)
        >>> q_fun.shape
        (9, 2, 4, 2)
        >>> state_env = np.array([3, 1, 2])
        >>> state_ext = np.array([0, 1, 0])
        >>> action_env = np.array([2, 3, 0])
        >>> action_ext = np.array([1, 0, 0])
        >>> value = q_fun(state_env, state_ext, action_env, action_ext)
        >>> value.shape
        (3,)
    """

    def __init__(self, *shape, init_value_min: float = 0.0, init_value_max: float = 0.0, **kwargs):
        """
        Args:
            shape (int...): a sequence of integers defining the shape of the table,
            init_value_min (float): the initial table is filled according to a
                uniform distribution in [init_value_min, init_value_max],
            init_value_max (float): see above,
        """

        self.shape = shape
        assert init_value_min <= init_value_max, "invalid initialization"
        self._init_value_min = init_value_min
        self._init_value_max = init_value_max
        self._table = np.random.uniform(init_value_min, init_value_max, shape)

    def __call__(self, *args):
        if not args:
            return self._table.copy()
        return self._table[tuple(np.stack(args))].copy()

    def update(self, *args, target=None):
        self._table[tuple(np.stack(args))] = target

    def reset(self):
        self._table = np.random.uniform(
            self._init_value_min,
            self._init_value_max,
            self.shape,
        )

    def copy_from(self, source):
        self._table = source._table.copy()

    def randomize(self, std: float = 1.0):
        self._table += np.random.randn(*self._table.shape) * std


class CountTable(Table):
    """
    The table counts input occurrences.
    It expects to receive one new observation at a time.
    """

    def update(self, *args):
        self._table[tuple(np.stack(args))] += 1

    def reset(self):
        self._table.fill(0)


class RunningMeanTable(Table):
    """
    The table stores the running mean of an observed variable (such as the reward).
    It expects to receive one new observation at a time.
    """

    def __init__(self, *shape, **kwargs):
        Table.__init__(self, *shape, **kwargs)
        self._count = np.zeros(self.shape)

    def update(self, *args, target=None):
        input = tuple(np.stack(args))
        self._count[input] += 1
        n = self._count[input]
        old_mean = self._table[input]
        new_mean = old_mean * (n - 1) / n + target / n
        self._table[input] = new_mean

    def reset(self):
        Table.reset(self)
        self._count.fill(0.0)

    def copy_from(self, source):
        Table.copy_from(self, source)
        self._count = source._count.copy()


class MSETable(Table):
    """
    It is updated using the MSE between the current table value (such as a Q-value)
    and a target (such as the TD target).
    """

    def update(self, *args, target=None, stepsize=1.0):
        """
        If there is more than one sample for the same input pair, the
        gradient is averaged. For example,

            >>> state = np.array([0, 0, 0])
            >>> action = np.array([2, 1, 2])
            >>> target = np.array([0.5, 0.2, 1.0])

        There are two samples for the state-action pair (0, 2), with targets
        0.5 and 1.0, respectively. The Q-value corresponding to this pair will
        be updated using the average of the gradient of those two samples.
        """

        input = np.stack(args)
        prediction = self._table[tuple(input)]
        loss = 0.5 * (target - prediction) ** 2
        gradient = target - prediction
        gradient *= stepsize

        unique_input = np.unique(input, axis=1)
        avg_gradient = np.zeros((unique_input.shape[1], *gradient.shape[1:]))
        for i, pair in enumerate(unique_input.T):
            idx = (pair[..., None] == input).sum(0) == input.shape[0]
            avg_gradient[i] = gradient[idx].mean(0)

        self._table[tuple(unique_input)] += avg_gradient

        return loss
