from itertools import product
from random import randint, choice, choices, sample, shuffle
from typing import List, Dict, Tuple, Any, Union

import numpy as np


IntegerPair = Tuple[int, int]


def uniform_integer(
    bounds: IntegerPair
) -> int:
    """
    random uniform integer

    bounds: (lower_bound, upper_bound)  Tuple
    """
    lower_bound, upper_bound = bounds
    return randint(lower_bound, upper_bound)


def randint_with_positive_bias(
    a: int,
    b: int
) -> int:
    """
    randomly uniform integer, but with bias towards upper bound (b)
    """
    return randint(randint(a, b), b)


def randint_with_negative_bias(
    a: int,
    b: int
) -> int:
    """
    randomly uniform integer, but with bias towards lower bound (a)
    """
    return randint(a, randint(a, b))


def within_euclidean_distance(
    a: IntegerPair,
    b: IntegerPair,
    d: int
) -> bool:
    """
    returns True iff a and b are at most manhattan distance d away from each other

    a: (i, j) location tuple
    b: (i, j) location tuple
    d: euclidean distance
    """
    a_i, a_j = a
    b_i, b_j = b
    return (((a_i - b_i) ** 2 + (a_j - b_j) ** 2) ** 0.5) <= d


def within_manhattan_distance(
    a: IntegerPair,
    b: IntegerPair,
    d: int
) -> bool:
    """
    returns True iff a and b are at most euclidean distance d away from each other

    a: (i, j) location tuple
    b: (i, j) location tuple
    d: manhattan distance
    """
    a_i, a_j = a
    b_i, b_j = b
    return (abs(a_i - b_i) + abs(a_j - b_j)) <= d


def within_chebyshev_distance(
    a: IntegerPair,
    b: IntegerPair,
    d: int
) -> bool:
    """
    returns True iff a and b are at most chebyshev distance d away from each other

    a: (i, j) location tuple
    b: (i, j) location tuple
    d: chebyshev distance
    """
    a_i, a_j = a
    b_i, b_j = b
    return max(abs(a_i - b_i), abs(a_j - b_j)) <= d


def dneighbors(
    loc: IntegerPair
) -> np.ndarray:
    """
    cells north, south, east and west

    loc: (row, column) tuple
    """
    i, j = loc
    return np.array([(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)])


def ineighbors(
    loc: IntegerPair
) -> np.ndarray:
    """
    cells north-east, south-east, north-west and south-west

    loc: (row, column) tuple
    """
    i, j = loc
    return np.array([(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)])


def neighbors(
    loc: IntegerPair
) -> np.ndarray:
    """
    all eight directly adjacent cells

    loc: (row, column) tuple
    """
    return np.concatenate((dneighbors(loc), ineighbors(loc)))


def square_view(
    d: int
) -> np.ndarray:
    """
    rectangular viewfield with radius d

    d: radius
    """
    arr = np.arange(-d, d + 1, 1)
    arr = np.array(list(product(arr, arr)))
    return arr[(arr[:, 0] != 0) | (arr[:, 1] != 0)]


def circle_view(
    d: int
) -> np.ndarray:
    """
    circular viewfield with radius d

    d: radius
    """
    arr = square_view(d)
    distances = np.sqrt(np.sum(np.power(arr, 2), axis=1))
    mask = distances <= d
    return arr[mask]


def is_int(
    x: Any
) -> bool:
    """
    whether or not an object is an integer
    """
    return isinstance(x, (np.integer, int))


def is_vec(
    x: Any
) -> bool:
    """
    whether or not an object is an array of two integers
    """
    if not (isinstance(x, tuple) or isinstance(x, list)):
        return False
    if len(x) != 2:
        return False
    if (not is_int(x[0])) or (not is_int(x[1])):
        return False
    return True


def vec_add(
    a: IntegerPair,
    b: IntegerPair
) -> IntegerPair:
    """
    vector addition
    """
    return (a[0] + b[0], a[1] + b[1])


DISTANCE_MAPPER = {
    'euclidean': within_euclidean_distance,
    'manhattan': within_manhattan_distance,
    'chebyshev': within_chebyshev_distance
}

DIRECS = list(set(map(tuple, neighbors((0, 0)))) - {(0, 0)})

VIEW_FUNCTION_MAPPER = {
    'square': square_view,
    'circle': circle_view
}
