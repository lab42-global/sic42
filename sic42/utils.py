import sys
import os
import copy
import json
import shutil
import importlib

import numpy as np

from itertools import combinations
from typing import List, Dict, Tuple
from sys import getsizeof



SUBREGION_PICKER = {
    'everywhere': lambda grid_shape, options: options,
    'top_half': lambda grid_shape, options: options[options[:, 0] < grid_shape[0] // 2],
    'bottom_half': lambda grid_shape, options: options[options[:, 0] > grid_shape[0] // 2],
    'left_half': lambda grid_shape, options: options[options[:, 1] < grid_shape[1] // 2],
    'right_half': lambda grid_shape, options: options[options[:, 1] > grid_shape[1] // 2],
    'top_left_quadrant': lambda grid_shape, options: options[(options[:, 0] < grid_shape[0] // 2) & (options[:, 1] < grid_shape[1] // 2)],
    'top_right_quadrant': lambda grid_shape, options: options[(options[:, 0] < grid_shape[0] // 2) & (options[:, 1] > grid_shape[1] // 2)],
    'bottom_left_quadrant': lambda grid_shape, options: options[(options[:, 0] > grid_shape[0] // 2) & (options[:, 1] < grid_shape[1] // 2)],
    'bottom_right_quadrant': lambda grid_shape, options: options[(options[:, 0] > grid_shape[0] // 2) & (options[:, 1] > grid_shape[1] // 2)],
}

AGENT_SELF_VIEW_ATTRIBUTES = [
    'swarm_name',
    'energy_level',
    'inbox',
    'memory_dict',
    'n_pheromone_symbols',
    'initial_pheromone_intensity',
    'view_distance',
    'existence_cost',
    'activity_points_per_tick',
    'remaining_activity_points',
    'max_memory_size',
    'carry_cost_factor',
    'max_weight_carriable',
    'max_broadcast_size',
    'attack_strength',
    'defense_strength',
    'max_write_size',
    'max_allowed_powerups',
    'max_allowed_upgrades',
    'num_past_upgrades'
]

NUMERICAL_AGENT_ATTRIUTES = [
    'view_distance',
    'energy_level',
    'existence_cost',
    'activity_points_per_tick',
    'n_pheromone_symbols',
    'max_memory_size',
    'carry_cost_factor',
    'max_weight_carriable',
    'max_broadcast_size',
    'attack_strength',
    'defense_strength',
    'max_write_size',
    'max_allowed_powerups',
    'max_allowed_upgrades',
    'num_past_upgrades',
]

POSITIVE_BIAS_MAPPER = {
    'view_distance': True,
    'energy_level': True,
    'existence_cost': False,
    'activity_points_per_tick': True,
    'max_memory_size': True,
    'n_pheromone_symbols': True,
    'initial_pheromone_intensity': True,
    'pheromone_intensity_decay': False,
    'carry_cost_factor': False,
    'max_weight_carriable': True,
    'max_broadcast_size': True,
    'attack_strength': True,
    'defense_strength': True,
    'max_write_size': True,
    'max_allowed_powerups': True,
    'max_allowed_upgrades': True
}


def to_json(
    self_view: Dict,
    relative_indices: np.array,
    entities: np.array,
    pheromones: np.array
) -> None:
    """
    make json serializable, save as json
    """
    json_obj = {
        'self_view': self_view,
        'relative_indices': relative_indices.tolist(),
        'entities': entities.tolist(),
        'pheromones': pheromones.tolist()
    }
    with open('agent_input.json', 'w') as fp:
        json.dump(json_obj, fp)


def get_competitors(
    folder: str
) -> Dict:
    """
    get map from competitor name to agent behavior module
    """
    competitors = dict()
    for fn in os.listdir(folder):
        if fn.endswith('.py'):
            teamname = fn.split('.')[0]
            try:
                spec = importlib.util.spec_from_file_location(teamname, os.path.join(folder, fn))
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                competitors[teamname] = module.main
            except Exception as err:
                print(f'Error on loading agent behavior module {fn}: "{err}"')
    return competitors


def generate_config(
    meta_config_path: str,
    teamnames: List[str]
) -> Dict:
    """
    construct configuration file from meta config
    """
    with open(meta_config_path, 'r') as fp:
        config = json.load(fp)
    config['environment']['background_symbol'] = 0
    swarm_meta_config = config['environment'].pop('swarms_meta_config')
    config['environment']['swarm_configs'] = dict()
    for i, teamname in enumerate(teamnames):
        swarm_config = copy.deepcopy(swarm_meta_config)
        swarm_config['symbol'] = i + 1
        swarm_config['agent_params']['behavior'] = teamname
        config['environment']['swarm_configs'][teamname] = swarm_config
    item_types = list(config['environment']['item_configs'])
    for i, item_type in enumerate(item_types):
        config['environment']['item_configs'][item_type]['symbol'] = len(teamnames) + i + 1
    time_multiplier = config['simulation'].pop('time_multiplier_in_seconds')
    expected_n_agents_per_swarm = (swarm_meta_config['n_agents_bounds'][0] + swarm_meta_config['n_agents_bounds'][1]) // 2
    n_agent_code_executions_proxy = expected_n_agents_per_swarm * config['simulation']['n_timesteps']
    config['simulation']['available_runtime_per_swarm'] = int(n_agent_code_executions_proxy * time_multiplier)
    return config


def getsizeofdict(
    d: Dict
) -> int:
    """
    return the size of a dictionary in bytes
    """
    s = 0
    for k, v in d.items():
        s += getsizeof(k)
        if isinstance(v, dict):
            s += getsizeofdict(v)
        else:
            s += getsizeof(v)
    return s


def delnodes(
    paths: List[str]
) -> None:
    """
    delete all folders including subfolders, if existent
    """
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        while os.path.exists(path):
            pass

