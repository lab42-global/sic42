import json

from random import randint, choice
from typing import List, Dict, Tuple, Any, Union

import numpy as np

import sys
sys.path.append('...')
from sic42.math_utils import *


def veccast(vec):
    return list(map(int, vec))


def from_json(path='agent_input.json'):
    with open(path, 'r') as fp:
        json_obj = json.load(fp)
    return (
        json_obj['self_view'],
        np.array(json_obj['relative_indices']),
        np.array(json_obj['entities']),
        np.array(json_obj['pheromones'], dtype='object')
    )


def get_closest(
    srcpos: np.ndarray,
    pool: np.ndarray
) -> np.ndarray:
    """
    returns entry from pool with closest manhattan distance to srcpos

    srcpos: reference position
    pool: array of positions
    """
    deltas = np.abs(pool - srcpos)
    distances = np.sum(deltas, axis=1)
    idx = np.argmin(distances)
    return pool[idx]


def main():
    """ focus on eating, but with some probability to reproduce """
    REPRODUCTION_PROBABILITY = 0.1
    self_view, indices, entities, pheromones = from_json()
    names = np.array([entity.get('name', None) for entity in entities])
    edibility = np.array([entity.get('edible', False) for entity in entities])
    empties = set(map(tuple, indices[names == 'empty'])) & set(DIRECS)
    desired_actions = []
    if sum(edibility) > 0:
        locs = indices[edibility]
        target_loc = get_closest(np.array([0, 0]), locs)
        direction = veccast(np.sign(target_loc))
        if within_chebyshev_distance(target_loc, (0, 0), 1):
            desired_actions.append(('eat', veccast(target_loc)))
        elif tuple(direction) in empties:
            if randint(0, 100) < REPRODUCTION_PROBABILITY * 100:
                desired_actions.append(('reproduce', veccast(direction)))
            else:
                desired_actions.append(('step', veccast(direction)))
    elif len(empties) > 0:
        direction = choice(list(empties))
        desired_actions.append(('step', veccast(direction)))    
    with open('agent_output.json', 'w') as fp:
        json.dump(desired_actions, fp)


