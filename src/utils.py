import numpy as np
import random


def random_argmax(x, rng_generator=np.random.Generator):
    """
    Simple random tiebreak for np.argmax() for when there are multiple max values.
    """

    best = np.argwhere(x == x.max())
    i = rng_generator.choice(range(best.shape[0]))
    return tuple(best[i])


def random_argmin(x, rng_generator=np.random.Generator):
    """
    Simple random tiebreak for np.argmin() for when there are multiple min values.
    """

    best = np.argwhere(x == x.min())
    i = rng_generator.choice(range(best.shape[0]))
    return tuple(best[i])


# https://en.wikipedia.org/wiki/Pairing_function
def cantor_pairing(x: int, y: int) -> int:
    """
    Cantor pairing function to uniquely encode two
    natural numbers into a single natural number.
    Used for seeding.

    Args:
        x (int): first number,
        y (int): second number,

    Returns:
        A unique integer computed from x and y.
    """

    return int(0.5 * (x + y) * (x + y + 1) + y)


def set_rng_seed(seed: int = None) -> None:
    """
    Set random number generator seed across modules
    with random/stochastic computations.

    Args:
        seed (int)
    """

    np.random.seed(seed)
    random.seed(seed)


def dict_to_id(d: dict) -> str:
    """
    Parse a dictionary and generate a unique id.
    The id will have the initials of every key followed by its value.
    Entries are separated by underscore.
    If a key's value is None, it will be skipped.

    Example:
        d = {"first_key": 0, "some_key": True, "another_key": None} -> fk0_skTrue
    """

    def make_prefix(key: str) -> str:
        return "".join(w[0] for w in key.split("_"))

    return "_".join([f"{make_prefix(k)}{v}" for k, v in d.items() if v is not None])


def pprint_4d(arr):
    """
    Utility to pretty print 4D np.array.
    Example:

    >>> arr = np.random.rand(2, 3, 4, 2)
    >>> pprint_4d(arr)

        [[0.8   0.239 0.776 0.59      | 0.338 0.383 0.996 0.508]
         [0.149 0.004 0.963 0.645     | 0.974 0.349 0.314 0.686]
         [    -     -     -     -     |     -     -     -     -]
         [0.917 0.438 0.875 0.876     | 0.684 0.976 0.993 0.904]
         [0.463 0.661 0.835 0.023     | 0.555 0.609 0.69  0.47 ]
         [    -     -     -     -     |     -     -     -     -]
         [0.997 0.155 0.027 0.001     | 0.44  0.783 0.678 0.272]
         [0.164 0.5   0.369 0.734     | 0.526 0.692 0.138 0.614]]

    Outer rows are for the 2nd dimension.
    Outer columns are for the 4th dimension.
    Inner matrices are for dimensions (1, 3).
    """

    a, b, c, d = arr.shape
    tmp = arr.reshape(a * b, c * d, order="F")
    tmp = np.insert(tmp, np.arange(a, a * b, a), np.nan, 0)
    tmp = np.insert(tmp, np.arange(c, c * d, c), np.inf, 1)

    with np.printoptions(
        precision=3,
        suppress=True,
        threshold=np.inf,
        linewidth=np.inf,
        infstr="|",
        nanstr="-",
        edgeitems=1000,
    ):
        print(tmp)


def mesh_combo(*args):
    """
    Simple meshgrid + flatten to get all combinations of many arrays.
    """

    return np.stack([i.flatten() for i in np.meshgrid(*args)]).T
