from abc import ABC, abstractmethod


class Parameter(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def step(self, **kwargs):
        pass

    @property
    def value(self):
        return self._value

    @abstractmethod
    def reset(self):
        pass


class LinearDecay(Parameter):
    """
    Linearly decaying parameter with coefficient 'alpha' and warm-up time.

        k_t = k_0 - max(t - t_warm, 0) * alpha

    Args:
        init_value (float): the initial value of k,
        min_value (float): the final value of K,
        warmup (int, 0): how many steps before decaying starts,
        decay (float, None): the decay value,
        steps (int, None): if decay is not specified, it will be calculated
            automatically with alpha = (init_value - min_value) / (steps - warmup),
    """

    def __init__(
        self,
        init_value: float = 1.0,
        min_value: float = 0.1,
        warmup: int = 0,
        decay: float = None,
        steps: int = None,
        **kwargs,
    ):
        assert warmup >= 0, "warmup must be non-negative"

        init_value = max(init_value, min_value)
        self._init_value = init_value
        self._min_value = min_value
        self._t = 0
        self._t_warm = warmup

        if steps is not None:
            self._decay = (init_value - min_value) / (steps - warmup)
        else:
            if decay is not None:
                self._decay = decay
            else:
                assert init_value == min_value, (
                    f"to decay from {init_value} to {min_value} "
                    f"you must specify either 'decay' or 'steps'"
                )
                self._decay = 1.0
        self._value = init_value

    def step(self):
        self._t += 1
        if self._t >= self._t_warm:
            self._value = max(self._value - self._decay, self._min_value)

    def reset(self):
        self._t = 0
        self._value = self._init_value

    @property
    def value(self):
        return self._value
