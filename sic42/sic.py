import importlib
import time
import shutil
import random
import json
import copy
import math
from collections.abc import Iterable
from typing import List, Dict, Tuple, Any, Union

import tqdm
import numpy as np
import pandas as pd

from sic42.utils import *
from sic42.math_utils import *
from sic42.visualization import *

from random import shuffle
from itertools import combinations


class Powerup:
    def __init__(
        self,
        attribute: str,
        delta: int,
        duration: int
    ):
        self.attribute = attribute
        self.delta = uniform_integer(delta)
        self.duration = uniform_integer(duration)

    def get_outside_view(
        self
    ) -> Dict:
        return {
            'attribute': self.attribute,
            'delta': self.delta,
            'duration': self.duration
        }
    
    def __str__(
        self
    ) -> str:
        return f'Powerup(attribute:{self.attribute},delta:{self.delta},duration:{self.duration})'


class Upgrade:
    def __init__(
        self,
        attribute: str,
        delta: int
    ):
        self.attribute = attribute
        self.delta = uniform_integer(delta)

    def get_outside_view(
        self
    ) -> Dict:
        return {
            'attribute': self.attribute,
            'delta': self.delta
        }
    
    def __str__(
        self
    ) -> str:
        return f'Upgrade(attribute:{self.attribute},delta:{self.delta})'


class Item:
    def __init__(
        self,
        name: str,
        loc: IntegerPair,
        item_id: int,
        energy_bounds: IntegerPair=(0, 0),
        max_energy_bounds: IntegerPair=[0, 0],
        energy_regrowth_bounds: IntegerPair=[0, 0],
        disappears_on_no_energy: bool=False,
        pickupable: bool=False,
        edible: bool=False,
        infostorage: bool=False,
        powerup: Dict=None,
        upgrade: Dict=None,
        weight_bounds: IntegerPair=[0, 0]
    ):
        self.name = name
        self.item_id = item_id
        self.loc = loc
        self.weight = uniform_integer(weight_bounds)
        self.energy = uniform_integer(energy_bounds)
        self.max_energy = uniform_integer(max_energy_bounds)
        self.energy_regrowth = uniform_integer(energy_regrowth_bounds)
        self.disappears_on_no_energy = disappears_on_no_energy
        self.edible = edible
        self.pickupable = pickupable
        self.infostorage = infostorage
        self.information = ''
        self.powerup = None
        self.upgrade = None
        if powerup is not None:
            powerup_spawn_probability = powerup['spawn_probability']
            powerup_config = {k: v for k, v in powerup.items() if k != 'spawn_probability'}
            if randint(0, 100) <= (powerup_spawn_probability * 100):
                self.powerup = Powerup(**powerup_config)
        if upgrade is not None:
            upgrade_spawn_probability = upgrade['spawn_probability']
            upgrade_config = {k: v for k, v in upgrade.items() if k != 'spawn_probability'}
            if randint(0, 100) <= (upgrade_spawn_probability * 100):
                self.upgrade = Upgrade(**upgrade_config)
    
    def get_outside_view(
        self
    ) -> Dict:
        outside_view = {
            'type': 'item',
            'name': self.name,
            'energy': self.energy,
            'pickupable': self.pickupable,
            'edible': self.edible,
            'infostorage': self.infostorage,
            'weight': self.weight
        }
        if self.powerup is not None:
            outside_view['powerup'] = self.powerup.get_outside_view()
        if self.upgrade is not None:
            outside_view['upgrade'] = self.upgrade.get_outside_view()
        return outside_view
    
    def item_hash(
        self
    ) -> str:
        keyvals = {k: str(v) for k, v in vars(self).items() if k not in ['item_id', 'loc', 'information']}
        srtd = sorted(list(keyvals.items()), key=lambda i: i[0])
        hashstr = ','.join([f'{k}:{v}' for k, v in srtd])
        return hashstr


class Action:
    def __init__(
        self,
        activity_points_cost_bounds: IntegerPair=[0, 0],
        energy_level_cost_bounds: IntegerPair=[0, 0],
        distance_metric: str='chebyshev',
        max_distance_bounds: IntegerPair=[1, 1],
        required_energy_level_bounds: IntegerPair=[0, 0]
    ):
        self.activity_points_cost = uniform_integer(activity_points_cost_bounds)
        self.energy_level_cost = uniform_integer(energy_level_cost_bounds)
        self.distance_metric = distance_metric
        self.max_distance = uniform_integer(max_distance_bounds)
        self.required_energy_level = uniform_integer(required_energy_level_bounds)


class Agent:
    def __init__(
        self,
        loc: IntegerPair=None,
        agent_id: int=None,
        swarm_name: str=None,
        view_distance_bounds: IntegerPair=None,
        view_function_indicator: str=None,
        initial_energy_level_bounds: IntegerPair=[1, 1],
        existence_cost_bounds: IntegerPair=[0, 0],
        activity_points_per_tick_bounds: IntegerPair=[0, 0],
        actions_config: Dict=None,
        carry_cost_factor_bounds: IntegerPair=[0, 0],
        max_weight_carriable_bounds: IntegerPair=[0, 0],
        max_memory_size_bounds: IntegerPair=[0, 0],
        behavior: str=None,
        attack_strength_bounds: IntegerPair=[0, 0],
        defense_strength_bounds: IntegerPair=[0, 0],
        max_broadcast_size_bounds: IntegerPair=[0, 0],
        n_pheromone_symbols_bounds: IntegerPair=[0, 0],
        initial_pheromone_intensity_bounds: IntegerPair=[0, 0],
        pheromone_intensity_decay_bounds: IntegerPair=[0, 0],
        max_write_size_bounds: IntegerPair=[0, 0],
        max_allowed_powerups_bounds: IntegerPair=[0, 0],
        max_allowed_upgrades_bounds: IntegerPair=[0, 0],
    ):
        self.agent_id = agent_id
        self.swarm_name = swarm_name
        self.view_distance = uniform_integer(view_distance_bounds)
        self.view_function_indicator = view_function_indicator
        self.energy_level = uniform_integer(initial_energy_level_bounds)
        self.loc = loc
        self.memory_dict = dict()
        self.existence_cost = uniform_integer(existence_cost_bounds)
        self.activity_points_per_tick = uniform_integer(activity_points_per_tick_bounds)
        self.remaining_activity_points = self.activity_points_per_tick
        self.max_memory_size = uniform_integer(max_memory_size_bounds)
        self.n_pheromone_symbols = uniform_integer(n_pheromone_symbols_bounds)
        self.initial_pheromone_intensity = uniform_integer(initial_pheromone_intensity_bounds)
        self.pheromone_intensity_decay = uniform_integer(pheromone_intensity_decay_bounds)
        self.behavior = behavior
        self.inventory = None
        self.inbox = []
        self.powerups = dict()

        self.carry_cost_factor = uniform_integer(carry_cost_factor_bounds)
        self.max_weight_carriable = uniform_integer(max_weight_carriable_bounds)
        self.max_broadcast_size = uniform_integer(max_broadcast_size_bounds)
        self.attack_strength = uniform_integer(attack_strength_bounds)
        self.defense_strength = uniform_integer(defense_strength_bounds)
        self.max_write_size = uniform_integer(max_write_size_bounds)
        self.max_allowed_powerups = uniform_integer(max_allowed_powerups_bounds)
        self.max_allowed_upgrades = uniform_integer(max_allowed_upgrades_bounds)
        self.num_past_upgrades = 0

        for action_name, action_params in actions_config.items():
            setattr(self, action_name, Action(**action_params))
    

    def self_view(
        self
    ) -> Dict:
        view = {'inventory_info': None}
        for attr in AGENT_SELF_VIEW_ATTRIBUTES:
            view[attr] = copy.deepcopy(getattr(self, attr))
        if (self.inventory is not None) and (self.inventory.infostorage):
            view['inventory_info'] = self.inventory.information
        view['memory'] = copy.deepcopy(view['memory_dict'])
        del view['memory_dict']
        return view
    

    def get_outside_view(
        self
    ) -> Dict:
        return {
            'type': 'swarm',
            'name': self.swarm_name
        }


class Environment:
    def __init__(
        self,
        seed: int,
        height_bounds: IntegerPair,
        width_bounds: IntegerPair,
        background_symbol: int,
        swarm_configs: Dict,
        item_configs: Dict,
        agent_behavior_mapper: Dict
    ):
        random.seed(seed)
        self.height = uniform_integer(height_bounds)
        self.width = uniform_integer(width_bounds)
        self.background_symbol = background_symbol
        self.swarm_configs = swarm_configs
        self.item_configs = item_configs
        self.grid = None
        self.pheromones = None
        self.empty = None
        self.swarms = dict()
        self.items = dict()
        self.next_item_id = None
        self.next_agent_id = None
        self.agent_behavior_mapper = agent_behavior_mapper
        self.next_item_id = 0
        self.next_agent_id = 0
    
    
    def get_n_empty_cells_in_subregion(
        self,
        num: int,
        subregion
    ) -> None:
        """ 
        get empty cells from subregion
        """
        options = np.array(np.where(self.grid == self.background_symbol)).T
        num = min(num, len(options))
        if isinstance(subregion, str):
            options = SUBREGION_PICKER[subregion](
                grid_shape=self.grid.shape, options=options
            )
        else:
            indicator = subregion['view_function_indicator']
            radius = subregion['radius']
            center = subregion['center']
            locs = VIEW_FUNCTION_MAPPER[indicator](radius) + np.array(center)
            options = np.array(list(set(map(tuple, locs)) & set(map(tuple, options))))
        locations = np.array(sample(list(options), num))
        return locations
    
    
    def initialize_items(
        self
    ) -> None:
        """
        for each item category, sample and construct items and add them to the environment
        """
        for item_name, item_config in self.item_configs.items():
            config = copy.deepcopy(item_config)
            num = uniform_integer(bounds=config.pop('n_items_bounds'))
            locations = self.get_n_empty_cells_in_subregion(
                num=num, subregion=config.pop('subregion')
            )
            self.grid[locations[:, 0], locations[:, 1]] = config.pop('symbol')
            locations = [tuple(loc) for loc in locations]
            self.items[item_name] = {
                loc: Item(
                    name=item_name, loc=loc, item_id=self.next_item_id+i, **config
                ) for i, loc in enumerate(locations)
            }
            self.next_item_id += num
    

    def initialize_swarms(
        self
    ) -> None:
        """
        for each swarm, sample and construct agents and add them to the environment
        """
        for swarm_name, swarm_config in self.swarm_configs.items():
            config = copy.deepcopy(swarm_config)
            num = uniform_integer(bounds=config['n_agents_bounds'])
            locations = self.get_n_empty_cells_in_subregion(
                num=num, subregion=config.pop('subregion')
            )
            self.grid[locations[:, 0], locations[:, 1]] = config['symbol']
            locations = [tuple(loc) for loc in locations]
            self.swarms[swarm_name] = {
                loc: Agent(
                    loc=loc, agent_id=self.next_agent_id+i, swarm_name=swarm_name,
                    **config['agent_params']
                ) for i, loc in enumerate(locations)
            }
            self.next_agent_id += num
    

    def respawn_items(
        self
    ) -> None:
        """
        for each item category, respawn items
        """
        for item_name, item_config in self.item_configs.items():
            config = copy.deepcopy(item_config)
            n_existing = len(self.items[item_name])
            min_n, max_n = config.pop('n_items_bounds')
            if n_existing >= min_n:
                continue
            num = uniform_integer((min_n - n_existing, max_n - n_existing))
            locations = self.get_n_empty_cells_in_subregion(
                num=num, subregion=config.pop('subregion')
            )
            self.grid[locations[:, 0], locations[:, 1]] = config.pop('symbol')
            locations = [tuple(loc) for loc in locations]
            self.items[item_name] = {
                **self.items[item_name],
                **{
                    loc: Item(
                        name=item_name, loc=loc, item_id=self.next_item_id+i, **config
                    ) for i, loc in enumerate(locations)
                }
            }
            self.next_item_id += num
    

    def initialize_board(
        self
    ) -> None:
        """
        construct grid, add items and swarms
        """
        shape = (self.height, self.width)
        self.grid = np.full(shape, self.background_symbol, dtype='uint8')
        self.pheromones = [[dict() for j in range(self.width)] for i in range(self.height)]
        self.initialize_items()
        self.initialize_swarms()
        empty_locs = np.array(np.where(self.grid == self.background_symbol)).T
        self.empty = set(map(tuple, empty_locs))
    
    
    def get_entity(
        self,
        loc: IntegerPair
    ):
        """
        returns the entity at a given location
        """
        for items in self.items.values():
            if loc in items:
                return items[loc]
        for swarm in self.swarms.values():
            if loc in swarm:
                return swarm[loc]


    def get_entity_view(
        self,
        loc: IntegerPair
    ) -> Dict:
        """
        returns the properties of an entity that should be viewable by an agent
        """
        entity = self.get_entity(loc)
        if entity is None:
            return {
                'type': None,
                'name': 'empty'
            }
        else:
            return copy.deepcopy(entity.get_outside_view())


class Simulation:
    def __init__(
        self,
        seed: int,
        environment: Environment,
        n_timesteps: int,
        available_runtime_per_swarm: int,
        stopping_criterion: str,
        pygame_params: Dict
    ):
        random.seed(seed)
        self.environment = environment
        self.n_timesteps = n_timesteps
        self.stopping_criterion = stopping_criterion
        self.frames = []
        self.metadata = []
        self.pheromone_frames = []
        self.base_views = dict()
        self.item_merging_lookup_table = dict()
        self.remaining_available_runtimes = {
            swarm_name: available_runtime_per_swarm for swarm_name in self.environment.swarms.keys()
        }
        self.pygame_params = pygame_params
        
        self.actions_mapper = {
            'eat': self.handle_eat,
            'pickup': self.handle_pickup,
            'putdown': self.handle_putdown,
            'step': self.handle_step,
            'memory': self.handle_memory,
            'broadcast': self.handle_broadcast,
            'attack': self.handle_attack,
            'reproduce': self.handle_reproduce,
            'pheromone': self.handle_pheromone,
            'infostorage': self.handle_infostorage,
            'powerup': self.handle_powerup,
            'upgrade': self.handle_upgrade,
            'itemmerge': self.handle_itemmerge
        }
        self.swarm_action_counters = {
            swarm_name: {
                action_name: 0 for action_name in self.actions_mapper.keys()
            } for swarm_name in self.environment.swarms.keys()
        }
        self.swarm_actions_log = {
            swarm_name: [] for swarm_name in self.environment.swarms.keys()
        }
        self.swarm_errors_log = {
            swarm_name: [] for swarm_name in self.environment.swarms.keys()
        }
        self.current_timestep = 0
        self.swarm_symbol_mapper = {
            swarm_name: config['symbol'] for swarm_name, config in self.environment.swarm_configs.items()
        }
        self.item_symbol_mapper = {
            item_name: config['symbol'] for item_name, config in self.environment.item_configs.items()
        }
    

    def merge_upgrades(self, upgrade_a, upgrade_b):
        if upgrade_a is None and upgrade_b is None:
            return None
        elif upgrade_a is not None and upgrade_b is not None:
            upgrade_src = choice((upgrade_a, upgrade_b))
        elif upgrade_a is not None and upgrade_b is None:
            upgrade_src = upgrade_a
        elif upgrade_a is None and upgrade_b is not None:
            upgrade_src = upgrade_b
        if POSITIVE_BIAS_MAPPER[upgrade_src.attribute]:
            delta = randint_with_positive_bias(a=upgrade_src.delta, b=upgrade_src.delta * 2)
        else:
            delta = randint_with_negative_bias(a=upgrade_src.delta // 2, b=upgrade_src.delta)
        return Upgrade(
            attribute=upgrade_src.attribute,
            delta=[delta, delta]
        )
        

    def merge_powerups(self, powerup_a, powerup_b):
        if powerup_a is None and powerup_b is None:
            return None
        elif powerup_a is not None and powerup_b is not None:
            powerup_src = choice((powerup_a, powerup_b))
        elif powerup_a is not None and powerup_b is None:
            powerup_src = powerup_a
        elif powerup_a is None and powerup_b is not None:
            powerup_src = powerup_b
        if POSITIVE_BIAS_MAPPER[powerup_src.attribute]:
            delta = randint_with_positive_bias(a=powerup_src.delta, b=powerup_src.delta * 2)
        else:
            delta = randint_with_negative_bias(a=powerup_src.delta // 2, b=powerup_src.delta)
        duration = randint_with_positive_bias(
            a=powerup_src.duration,
            b=powerup_src.duration * 2
        )
        return Powerup(
            attribute=powerup_src.attribute,
            delta=[delta, delta],
            duration=[duration, duration]
        )
        
    def merge_items(self, item_a, item_b):
        ab_hash = frozenset({item_a.item_hash(), item_b.item_hash()})
        if ab_hash not in self.item_merging_lookup_table:
            self.item_merging_lookup_table[ab_hash] = Item(
                name='_'.join(sorted(set([item_a.name, item_b.name]))),
                loc=None,
                item_id=None,
                weight_bounds=[randint_with_negative_bias(min(item_a.weight, item_b.weight), item_a.weight + item_b.weight)] * 2,
                energy_bounds=[randint_with_positive_bias(min(item_a.energy, item_b.energy), item_a.energy + item_b.energy)] * 2,
                max_energy_bounds=[randint_with_positive_bias(min(item_a.max_energy, item_b.max_energy), item_a.max_energy + item_b.max_energy)] * 2,
                energy_regrowth_bounds=[randint_with_positive_bias(min(item_a.energy_regrowth, item_b.energy_regrowth), item_a.energy_regrowth + item_b.energy_regrowth)] * 2,
                disappears_on_no_energy=choices(population=sorted([item_a.disappears_on_no_energy, item_b.disappears_on_no_energy]), weights=[2, 1])[0],
                edible=choices(population=sorted([item_a.edible, item_b.edible]), weights=[1, 2])[0],
                pickupable=choices(population=sorted([item_a.pickupable, item_b.pickupable]), weights=[1, 2])[0],
                infostorage=choices(population=sorted([item_a.infostorage, item_b.infostorage]), weights=[1, 2])[0],
                powerup=self.merge_powerups(item_a.powerup, item_b.powerup),
                upgrade=self.merge_upgrades(item_a.upgrade, item_b.upgrade)
            )
        merged_item = copy.deepcopy(self.item_merging_lookup_table[ab_hash])
        merged_item.item_id = self.environment.next_item_id
        self.environment.next_item_id += 1
        return merged_item
    
    
    def handle_step(
        self,
        step: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        check if desired step is valid, and if so, execute and handle side effects
        """
        if not hasattr(agent, 'step'):
            return False, 'agent does not have the ability to perform action'
        if not is_vec(step):
            return False, 'action value is not a valid (x, y) location'
        step_loc = vec_add(step, agent.loc)
        if step_loc not in self.environment.empty:
            return False, 'desired location is not empty'
        if not DISTANCE_MAPPER[agent.step.distance_metric](
            a=step_loc, b=agent.loc, d=agent.step.max_distance
        ):
            return False, 'desired location is not within reach'
        if agent.remaining_activity_points < agent.step.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if agent.energy_level < agent.step.required_energy_level:
            return False, 'insufficient energy level'
        agent.remaining_activity_points -= agent.step.activity_points_cost
        previous_loc = agent.loc
        agent.loc = step_loc
        self.environment.swarms[agent.swarm_name][agent.loc] = self.environment.swarms[agent.swarm_name].pop(previous_loc)
        self.environment.grid[agent.loc] = self.environment.swarm_configs[agent.swarm_name]['symbol']
        self.environment.empty.add(previous_loc)
        self.environment.grid[previous_loc] = self.environment.background_symbol
        return True, 'success'
    
    
    def handle_eat(
        self,
        to_eat: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        check if desired eating is valid, and if so, execute and handle side effects
        """ 
        if not hasattr(agent, 'eat'):
            return False, 'agent does not have the ability to perform action'
        if not is_vec(to_eat):
            return False, 'action value is not a valid (x, y) location'
        eat_loc = vec_add(to_eat, agent.loc)
        entity = self.environment.get_entity(eat_loc)
        if entity is None:
            return False, 'desired location is empty'
        if not isinstance(entity, Item):
            return False, 'desired location does not contain an item'
        if not entity.edible:
            return False, 'item is not edible'
        if not DISTANCE_MAPPER[agent.eat.distance_metric](
            a=eat_loc, b=agent.loc, d=agent.eat.max_distance
        ):
            return False, 'desired location is not within reach'
        if agent.remaining_activity_points < agent.eat.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if agent.energy_level < agent.eat.required_energy_level:
            return False, 'insufficient energy level'
        agent.energy_level += entity.energy
        entity.energy = 0
        agent.remaining_activity_points -= agent.eat.activity_points_cost
        if entity.disappears_on_no_energy and entity.energy == 0:
            self.environment.grid[eat_loc] = self.environment.background_symbol
            del self.environment.items[entity.name][eat_loc]
            self.environment.empty.add(eat_loc)
        return True, 'success'
    
    
    def handle_powerup(
        self,
        to_consume_loc: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        check if desired powerup is valid, and if so, execute and handle side effects
        """
        if not hasattr(agent, 'powerup'):
            return False, 'agent does not have the ability to perform action'
        if not is_vec(to_consume_loc):
            return False, 'action value is not a valid (x, y) location'
        true_to_consume_loc = vec_add(to_consume_loc, agent.loc)
        entity = self.environment.get_entity(true_to_consume_loc)
        if entity is None:
            return False, 'desired location is empty'
        if not isinstance(entity, Item):
            return False, 'desired location does not contain an item'
        if entity.powerup is None:
            return False, 'item does not have any powerup'
        if not DISTANCE_MAPPER[agent.powerup.distance_metric](
            a=true_to_consume_loc, b=agent.loc, d=agent.powerup.max_distance
        ):
            return False, 'desired location is not within reach'
        if agent.remaining_activity_points < agent.powerup.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if not hasattr(agent, entity.powerup.attribute):
            return False, f'agent does not have attribute {entity.powerup.attribute}'
        if len(agent.powerups) == agent.max_allowed_powerups:
            return False, 'agent has reached limit of maximum allowed powerups'
        if entity.powerup.attribute in agent.powerups:
            return False, 'agent is already using a powerup affecting that attribute'
        if agent.energy_level < agent.powerup.required_energy_level:
            return False, 'insufficient energy level'
        agent.remaining_activity_points -= agent.powerup.activity_points_cost
        previous = getattr(agent, entity.powerup.attribute)
        new = previous + entity.powerup.delta
        setattr(agent, entity.powerup.attribute, new)
        agent.powerups[entity.powerup.attribute] = (previous, entity.powerup.duration)
        entity.powerup = None
        return True, 'success'
    

    def handle_upgrade(
        self,
        to_consume_loc: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        check if desired upgrade is valid, and if so, execute and handle side effects
        """
        if not hasattr(agent, 'upgrade'):
            return False, 'agent does not have the ability to perform action'
        if not is_vec(to_consume_loc):
            return False, 'action value is not a valid (x, y) location'
        true_to_consume_loc = vec_add(to_consume_loc, agent.loc)
        entity = self.environment.get_entity(true_to_consume_loc)
        if entity is None:
            return False, 'desired location is empty'
        if not isinstance(entity, Item):
            return False, 'desired location does not contain an item'
        if entity.upgrade is None:
            return False, 'item does not have any upgrade'
        if not DISTANCE_MAPPER[agent.upgrade.distance_metric](
            a=true_to_consume_loc, b=agent.loc, d=agent.upgrade.max_distance
        ):
            return False, 'desired location is not within reach'
        if agent.remaining_activity_points < agent.upgrade.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if not hasattr(agent, entity.upgrade.attribute):
            return False, f'agent does not have attribute {entity.upgrade.attribute}'
        if agent.num_past_upgrades == agent.max_allowed_upgrades:
            return False, 'agent has reached limit of maximum allowed upgrades'
        if agent.energy_level < agent.upgrade.required_energy_level:
            return False, 'insufficient energy level'
        agent.remaining_activity_points -= agent.upgrade.activity_points_cost
        agent.num_past_upgrades += 1
        previous = getattr(agent, entity.upgrade.attribute)
        new = previous + entity.upgrade.delta
        setattr(agent, entity.upgrade.attribute, new)
        entity.upgrade = None
        return True, 'success'
    

    def handle_pickup(
        self,
        to_pickup: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        check if desired item pickup is valid, and if so, execute and handle side effects
        """
        if not hasattr(agent, 'pickup'):
            return False, 'agent does not have the ability to perform action'
        if not is_vec(to_pickup):
            return False, 'action value is not a valid (x, y) location'
        pickup_loc = vec_add(to_pickup, agent.loc)
        item_exists = False
        for name, items in self.environment.items.items():
            if pickup_loc in items.keys():
                item_exists = True
                item_type = name
                break
        if not item_exists:
            return False, 'item does not exist'
        if not self.environment.items[item_type][pickup_loc].pickupable:
            return False, 'item is not pickupable'
        if not DISTANCE_MAPPER[agent.pickup.distance_metric](
            a=pickup_loc, b=agent.loc, d=agent.pickup.max_distance
        ):
            return False, 'desired location is not within reach'
        if agent.remaining_activity_points < agent.pickup.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if self.environment.items[item_type][pickup_loc].weight > agent.max_weight_carriable:
            return False, 'item is too heavy to be picked up'
        if agent.inventory is not None:
            return False, 'agent inventory is already full'
        if agent.energy_level < agent.pickup.required_energy_level:
            return False, 'insufficient energy level'
        item = self.environment.items[item_type].pop(pickup_loc)
        item.loc = None
        agent.inventory = item
        agent.remaining_activity_points -= agent.pickup.activity_points_cost
        self.environment.grid[pickup_loc] = self.environment.background_symbol
        self.environment.empty.add(pickup_loc)
        return True, 'success'
    
    
    def handle_putdown(
        self,
        to_putdown: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        check if desired item putdown is valid, and if so, execute and handle side effects
        """
        if not hasattr(agent, 'putdown'):
            return False, 'agent does not have the ability to perform action'
        if not is_vec(to_putdown):
            return False, 'action value is not a valid (x, y) location'
        putdown_loc = vec_add(to_putdown, agent.loc)
        if putdown_loc not in self.environment.empty:
            return False, 'desired location is not empty'
        if not DISTANCE_MAPPER[agent.putdown.distance_metric](
            a=putdown_loc, b=agent.loc, d=agent.putdown.max_distance
        ):
            return False, 'desired location is not within reach'
        if agent.remaining_activity_points < agent.putdown.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if agent.inventory is None:
            return False, 'agent inventory is empty'
        if agent.energy_level < agent.putdown.required_energy_level:
            return False, 'insufficient energy level'
        item = agent.inventory
        item.loc = putdown_loc
        self.environment.items[item.name][putdown_loc] = item
        self.environment.empty.remove(putdown_loc)
        agent.inventory = None
        agent.remaining_activity_points -= agent.putdown.activity_points_cost
        self.environment.grid[putdown_loc] = self.environment.item_configs[item.name]['symbol']
        return True, 'success'


    def handle_broadcast(
        self,
        broadcast,
        agent: Agent
    ) -> bool:
        """
        check if desired broadcast is valid, and if so, broadcast
        """
        if not hasattr(agent, 'broadcast'):
            return False, 'agent does not have the ability to perform action'
        if not isinstance(broadcast, Iterable):
            return False, 'action value is not a list'
        for tup in broadcast:
            if not isinstance(tup, Iterable):
                return False, 'broadcast is not a list [loc, msg]'
            if len(tup) != 2:
                return False, 'broadcast is not a list [loc, msg]'
            loc, msg = tup
            if not is_vec(loc):
                return False, 'broadcast location is not a valid (x, y) location'
            entity = self.environment.get_entity(vec_add(loc, agent.loc))
            if type(entity) != Agent:
                return False, 'broadcast target is not an agent'
            if entity.swarm_name != agent.swarm_name:
                return False, 'broadcast target agent is from a different swarm'
            if not isinstance(msg, str):
                return False, 'broadcast message is not a string'
            if len(msg) > agent.max_broadcast_size:
                return False, 'broadcast messgae is too long'
            if not DISTANCE_MAPPER[agent.broadcast.distance_metric](vec_add(loc, agent.loc), agent.loc, agent.broadcast.max_distance):
                return False, 'broadcast target agent is not within reach'
        if agent.remaining_activity_points < len(broadcast) * agent.broadcast.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if agent.energy_level < agent.broadcast.required_energy_level:
            return False, 'insufficient energy level'
        agent.remaining_activity_points -= len(broadcast) * agent.broadcast.activity_points_cost
        for loc, msg in broadcast:
            self.environment.get_entity(vec_add(loc, agent.loc)).inbox.append(msg)
        return True, 'success'
        
    
    def handle_attack(
        self,
        victim_loc: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        check if attack is valid, and if so, attack
        """
        if not hasattr(agent, 'attack'):
            return False, 'agent does not have the ability to perform action'
        if not is_vec(victim_loc):
            return False, 'action value is not a valid (x, y) location'
        true_victim_loc = vec_add(victim_loc, agent.loc)
        victim = self.environment.get_entity(true_victim_loc)
        if type(victim) != Agent:
            return False, 'target is not an agent'
        if agent.remaining_activity_points < agent.attack.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if not DISTANCE_MAPPER[agent.attack.distance_metric](
            a=true_victim_loc,
            b=agent.loc,
            d=agent.attack.max_distance
        ):
            return False, 'desired location is not within reach'
        if agent.energy_level < agent.attack.required_energy_level:
            return False, 'insufficient energy level'
        damage = max(0, agent.attack_strength - victim.defense_strength)
        victim.energy_level -= damage
        agent.remaining_activity_points -= agent.attack.activity_points_cost
        if victim.energy_level <= 0:
            del self.environment.swarms[victim.swarm_name][true_victim_loc]
            self.environment.grid[true_victim_loc] = self.environment.background_symbol
        return True, 'success'
    

    def handle_reproduce(
        self,
        child_loc: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        checks if agent can reproduce, and if so, make child
        """
        if not hasattr(agent, 'reproduce'):
            return False, 'agent does not have the ability to perform action'
        if not is_vec(child_loc):
            return False, 'action value is not a valid (x, y) location'
        true_child_loc = vec_add(child_loc, agent.loc)
        if true_child_loc not in self.environment.empty:
            return False, 'desired location is not empty'
        if agent.remaining_activity_points < agent.reproduce.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if agent.energy_level < agent.reproduce.required_energy_level:
            return False, 'insufficient energy level'
        if not DISTANCE_MAPPER[agent.reproduce.distance_metric](
            a=true_child_loc,
            b=agent.loc,
            d=agent.reproduce.max_distance
        ):
            return False, 'desired location is not within reach'
        agent.remaining_activity_points -= agent.reproduce.activity_points_cost
        self.environment.empty.remove(true_child_loc)
        child = copy.deepcopy(agent)
        child.agent_id = self.environment.next_agent_id
        child.loc = true_child_loc
        half_energy_level = agent.energy_level // 2
        child.energy_level = half_energy_level
        agent.energy_level = half_energy_level
        self.deactivate_powerups(child)
        child.memory_dict = dict()
        child.inventory = None
        self.environment.next_agent_id += 1
        self.environment.grid[child.loc] = self.environment.swarm_configs[child.swarm_name]['symbol']
        self.environment.swarms[child.swarm_name][child.loc] = child
        return True, 'success'
    

    def handle_pheromone(
        self,
        pheromone: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        pheromone placement
        """
        if not hasattr(agent, 'pheromone'):
            return False, 'agent does not have the ability to perform action'
        if not isinstance(pheromone, Iterable):
            return False, 'action value is not of form (loc, symbol)'
        if len(pheromone) != 2:
            return False, 'action value is not of form (loc, symbol)'
        pheromone_loc, pheromone_symbol = pheromone
        if not isinstance(pheromone_symbol, int):
            return False, 'desired pheromone symbol is not an integer'
        if not (1 <= pheromone_symbol <= agent.n_pheromone_symbols):
            return False, 'desired pheromone symbol is not within valid range'
        if not is_vec(pheromone_loc):
            return False, 'desired location is not of form (x, y)'
        true_pheromone_loc = vec_add(pheromone_loc, agent.loc)
        i, j = true_pheromone_loc
        if not 0 <= i < self.environment.height:
            return False, 'desired location is not on the grid'
        if not 0 <= j < self.environment.width:
            return False, 'desired location is not on the grid'
        if agent.remaining_activity_points < agent.pheromone.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if agent.energy_level < agent.pheromone.required_energy_level:
            return False, 'insufficient energy level'
        if not DISTANCE_MAPPER[agent.pheromone.distance_metric](
            a=true_pheromone_loc,
            b=agent.loc,
            d=agent.pheromone.max_distance
        ):
            return False, 'desired location is not within reach'
        agent.remaining_activity_points -= agent.pheromone.activity_points_cost
        if pheromone_symbol not in self.environment.pheromones[i][j]:
            self.environment.pheromones[i][j][pheromone_symbol] = set()
        self.environment.pheromones[i][j][pheromone_symbol].add(
            (agent.initial_pheromone_intensity, agent.pheromone_intensity_decay)
        )
        return True, 'success'
    

    def handle_infostorage(
        self,
        infostorage: Tuple[IntegerPair, str],
        agent: Agent
    ) -> bool:
        """
        storing information in items
        """
        if not hasattr(agent, 'infostorage'):
            return False, 'agent does not have the ability to perform action'
        if not isinstance(infostorage, Iterable):
            return False, 'action value is not of the form (loc, info)'
        if len(infostorage) != 2:
            return False, 'action value is not of the form (loc, info)'
        write_loc, information = infostorage
        if not is_vec(write_loc):
            return False, 'action value is not a valid (x, y) location'
        true_write_loc = vec_add(write_loc, agent.loc)
        if not isinstance(information, str):
            return False, 'desired information is not a string'
        if len(information) > agent.max_write_size:
            return False, 'desired information is too long'
        entity = self.environment.get_entity(true_write_loc)
        if entity is None:
            return False, 'desired location does not contain an item'
        if not entity.infostorage:
            return False, 'desired item does not allow storing information'
        if agent.remaining_activity_points < agent.infostorage.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        if agent.energy_level < agent.infostorage.required_energy_level:
            return False, 'insufficient energy level'
        if not DISTANCE_MAPPER[agent.infostorage.distance_metric](
            a=true_write_loc,
            b=agent.loc,
            d=agent.infostorage.max_distance
        ):
            return False, 'desired location is not within reach'
        agent.remaining_activity_points -= agent.infostorage.activity_points_cost
        entity.information = information
        return True, 'success'
    

    def handle_itemmerge(
        self,
        merge_loc: IntegerPair,
        agent: Agent
    ) -> bool:
        """
        item merging
        """
        if not hasattr(agent, 'itemmerge'):
            return False, 'agent does not have the ability to perform action'
        if not is_vec(merge_loc):
            return False, 'action value is not a valid (x, y) location'
        true_merge_loc = vec_add(merge_loc, agent.loc)
        if not DISTANCE_MAPPER[agent.itemmerge.distance_metric](
            a=true_merge_loc, b=agent.loc, d=agent.itemmerge.max_distance
        ):
            return False, 'desired location is not within reach'
        if agent.remaining_activity_points < agent.itemmerge.activity_points_cost:
            return False, 'insufficient number of remaining activity points'
        entity = self.environment.get_entity(true_merge_loc)
        if entity is None:
            return False, 'desired location does not contain an agent'
        if not isinstance(entity, Agent):
            return False, 'desired location does not contain an agent'
        if entity.swarm_name != agent.swarm_name:
            return False, 'desired agent is of different swarm'
        if agent.inventory is None:
            return False, 'agent inventory is empty'
        if entity.inventory is None:
            return False, 'inventory of other agent is empty'
        if agent.energy_level < agent.itemmerge.required_energy_level:
            return False, 'insufficient energy level'
        agent.inventory = self.merge_items(agent.inventory, entity.inventory)
        agent.remaining_activity_points -= agent.itemmerge.activity_points_cost
        entity.inventory = None
        return True, 'success'
    

    def handle_memory(
        self,
        memory: Dict,
        agent: Agent
    ) -> bool:
        """
        check if desired memory update is valid, and if so, update agent memory
        """
        if not hasattr(agent, 'memory'):
            return False, 'agent does not have the ability to perform action'
        if not isinstance(memory, dict):
            return False, 'memory is not a dictionary'
        if getsizeofdict(memory) > agent.max_memory_size:
            return False, 'memory is too big'
        agent.memory_dict = memory
        return True, 'success'
    

    def handle_agent_post_action_updates(
        self,
        agent: Agent
    ) -> None:
        """
        handle agent energy level and let it die if it runs out of energy
        """
        agent.energy_level -= agent.existence_cost
        if agent.inventory is not None:
            agent.energy_level -= agent.inventory.weight * agent.carry_cost_factor
        agent.remaining_activity_points = agent.activity_points_per_tick
        agent.inbox = []
        powerups = dict()
        for attribute, (previous, remaining) in agent.powerups.items():
            remaining = remaining - 1
            if remaining == 0:
                setattr(agent, attribute, previous)
            else:
                powerups[attribute] = (previous, remaining)
        agent.powerups = powerups
        if agent.energy_level <= 0:
            self.environment.grid[agent.loc] = self.environment.background_symbol
            self.environment.empty.add(agent.loc)
            del self.environment.swarms[agent.swarm_name][agent.loc]
    
    def deactivate_powerups(
        self,
        agent: Agent
    ) -> None:
        """
        deactivate powerups
        """
        for attribute, (previous, remaining) in agent.powerups.items():
            setattr(agent, attribute, previous)
        agent.powerups = dict()
    
    def init_base_views(
        self
    ) -> None:
        """
        for each specified viewfield type and view distance, construct the field of view as seen from the origin
        """
        view_distance_lower_bound = min([config['agent_params']['view_distance_bounds'][0] for config in self.environment.swarm_configs.values()])
        view_distance_upper_bound = max([config['agent_params']['view_distance_bounds'][1] for config in self.environment.swarm_configs.values()])
        view_functions = set([config['agent_params']['view_function_indicator'] for config in self.environment.swarm_configs.values()])
        base_view_rng = range(view_distance_lower_bound, view_distance_upper_bound + 1)
        self.base_views = {
            view_function_name: {d: VIEW_FUNCTION_MAPPER[view_function_name](d) for d in base_view_rng} \
                for view_function_name in view_functions
        }
    
    def get_random_agent_order(
        self
    ) -> List[Tuple[str, IntegerPair]]:
        """
        random traversal order over all agents of all swarms
        """
        all_agents = []
        for swarm in self.environment.swarms.values():
            for agent in swarm.values():
                all_agents.append((agent.swarm_name, agent.loc))
        shuffle(all_agents)
        return all_agents
    
    
    def regrow_item_energy_levels(
        self
    ) -> None:
        """ 
        regrow item energies
        """
        for item_name, items in self.environment.items.items():
            for loc, item in items.items():
                item.energy = min(item.max_energy, item.energy + item.energy_regrowth)
    
    def decay_pheromones(
        self
    ) -> None:
        """
        decay pheromone intensities
        """
        h = self.environment.height
        w = self.environment.width
        new_pheromones = [[dict() for j in range(w)] for i in range(h)]
        for i, r in enumerate(self.environment.pheromones):
            for j, f in enumerate(r):
                new_pheromones[i][j] = {
                    s: {(l-d, d) for l, d in ld if l - d > 0} for s, ld in f.items()
                }
                new_pheromones[i][j] = {k: v for k, v in new_pheromones[i][j].items() if len(v) > 0}
        self.environment.pheromones = new_pheromones
    
    
    def compute_metadata(
        self
    ) -> None:
        """
        compute and save statistics on a swarm level for later plotting
        """
        frame_metadata = dict()
        for swarm_name, swarm in self.environment.swarms.items():
            frame_metadata[swarm_name] = dict()
            for aciton_name in self.actions_mapper.keys():
                action_count = self.swarm_action_counters[swarm_name][aciton_name]
                frame_metadata[swarm_name][f'{aciton_name} count'] = action_count
            for attribute_name in NUMERICAL_AGENT_ATTRIUTES:
                attribute_values = [getattr(agent, attribute_name) for agent in swarm.values()]
                if len(attribute_values) == 0:
                    frame_metadata[swarm_name][f'average {attribute_name}'] = 0
                else:
                    frame_metadata[swarm_name][f'average {attribute_name}'] = np.mean(attribute_values)
        self.metadata.append(frame_metadata)
    

    def handle_actions(
        self,
        desired_actions: Iterable,
        agent: Agent
    ) -> None:
        """
        handle each of the desired actions an agent would like to do
        """
        if not isinstance(desired_actions, Iterable):
            self.swarm_errors_log[agent.swarm_name].append({
                'timestep': self.current_timestep,
                'agent_id': agent.agent_id,
                'error_message': 'desired actions is not a list'
            })
            return
        for desired_action in desired_actions:
            succ = True
            if not isinstance(desired_action, Iterable):
                succ = False
                msg = f'desired action "{desired_action}" is not a list'
            if succ and len(desired_action) != 2:
                succ = False
                msg = f'desired action "{desired_action}" not of the form [action_name, action_value]'
            if succ:
                action_indicator, action_value = desired_action
            else:
                action_indicator, action_value = None, None
            if succ and action_indicator not in self.actions_mapper:
                succ = False
                msg = 'action name is not valid'
            if succ:
                handler = self.actions_mapper[action_indicator]
                succ, msg = handler(action_value, agent)
            if succ:
                self.swarm_action_counters[agent.swarm_name][action_indicator] += 1
            else:
                agent.remaining_activity_points -= 1
            if msg == 'success':
                msg = None
            action_log = {
                'action_name': action_indicator,
                'action_value': action_value,
                'successful': succ,
                'error_message': msg,
                'agent_id': agent.agent_id,
                'timestep': self.current_timestep
            }
            self.swarm_actions_log[agent.swarm_name].append(action_log)

    
    def get_viewfield(
        self,
        agent: Agent
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        returns (indices, values) tuple of stuff viewable by agent

        """
        agent_loc = np.array(agent.loc)
        locs = self.base_views[agent.view_function_indicator][agent.view_distance] + agent_loc
        h, w = self.environment.grid.shape
        row_valid = (locs[:, 0] >= 0) & (locs[:, 0] < h)
        col_valid = (locs[:, 1] >= 0) & (locs[:, 1] < w)
        valid = row_valid & col_valid
        viewable_absolute = locs[valid]
        entities = np.array([
            self.environment.get_entity_view(tuple(loc)) for loc in viewable_absolute
        ])
        viewable_relative = viewable_absolute - agent_loc
        pheromones = []
        for i, j in viewable_absolute:
            pheromones.append(list(self.environment.pheromones[i][j].keys()))
        pheromones = np.array(pheromones, dtype='object')
        return viewable_relative, entities, pheromones

    
    def handle_tick_info_for_visualizations(
        self
    ):
        self.frames.append(np.copy(self.environment.grid))
        self.pheromone_frames.append(copy.deepcopy(self.environment.pheromones))
        self.compute_metadata()
    

    def agent_turn(
        self,
        agent: Agent
    ):
        if self.remaining_available_runtimes[agent.swarm_name] < 0:
            self.swarm_errors_log[agent.swarm_name].append({
                'timestep': self.current_timestep,
                'agent_id': agent.agent_id,
                'error_message': 'no remaining time left'
            })
            return
        relative_indices, entities, pheromones = self.get_viewfield(agent)
        self_view = agent.self_view()
        to_json(self_view, relative_indices, entities, pheromones)
        desired_actions = []
        start = time.time()
        try:
            self.environment.agent_behavior_mapper[agent.behavior]()
        except Exception as err:
            self.swarm_errors_log[agent.swarm_name].append({
                'timestep': self.current_timestep,
                'agent_id': agent.agent_id,
                'error_message': f'error during execution of code: {err}'
            })
        end = time.time()
        turn_duration = end - start
        self.remaining_available_runtimes[agent.swarm_name] -= turn_duration
        try:
            with open('agent_output.json', 'r') as fp:
                desired_actions = json.load(fp)
        except Exception as err:
            self.swarm_errors_log[agent.swarm_name].append({
                'timestep': self.current_timestep,
                'agent_id': agent.agent_id,
                'error_message': f'error during reading of "agent_output.json": {err}'
            })
            desired_actions = []
        self.handle_actions(desired_actions, agent)
        for filename in ['agent_input.json', 'agent_output.json']:
            if os.path.isfile(filename):
                os.remove(filename)


    def simulate(
        self,
        desc: str='simulating'
    ) -> None:
        """
        main function for running the simulation
        """
        self.handle_tick_info_for_visualizations()
        self.init_base_views()
        for k in tqdm.tqdm(range(self.n_timesteps), desc=desc):
            self.current_timestep = k + 1
            all_agents = self.get_random_agent_order()
            for swarm_name, agent_location in all_agents:
                if agent_location not in self.environment.swarms[swarm_name]:
                    continue
                agent = self.environment.swarms[swarm_name][agent_location]
                self.agent_turn(agent)
                self.handle_agent_post_action_updates(agent)
            self.regrow_item_energy_levels()
            self.environment.respawn_items()
            if self.environment.pheromones is not None:
                self.decay_pheromones()
            self.handle_tick_info_for_visualizations()
            if self.stopping_criterion == 'all_agents_dead':
                n_agents_left = sum([len(swarm) for swarm in self.environment.swarms.values()])
                if n_agents_left == 0:
                    break
    
    def get_all_agents(
        self
    ) -> List[Agent]:
        """
        get all agents in the environment
        """
        return [agent for swarm in self.environment.swarms.values() for agent in swarm.values()]

    def get_all_items(
        self
    ) -> List[Item]:
        """
        get all items in the environment
        """
        return [item for items in self.environment.items.values() for item in items.values()]
        
    def get_pheromones(
        self
    ) -> List[Dict]:
        """
        get all pheromones in the environment
        """
        return self.environment.pheromones
    
    def get_basic_stats(
        self,
        stats
    ) -> Dict:
        """
        get basic stats about the environment to test graphing
        """
        return stats + [
            {
                swarm_name: {
                    'average agent energy': swarm_stats['average energy_level'],
                    'swarm size': len(self.environment.swarms[swarm_name])
                } for swarm_name, swarm_stats in self.metadata[-1].items()
            }
        ]
    
    def simulate_pygame(
        self,
        desc: str='simulating'
    ) -> None:
        """
        main function for running the simulation using pygame as a visualization tool
        """   
        stats = []
        show_video=True
        show_graphs=True
        scale=8
        visualizer = PygameVisualizer(
            width=self.environment.width,
            height=self.environment.height,
            swarm_symbol_mapper=self.swarm_symbol_mapper,
            item_symbol_mapper=self.item_symbol_mapper,
            scale=scale,
            record_video=show_video,
            show_graphs=show_graphs,
            pygame_params=self.pygame_params
        )
        self.handle_tick_info_for_visualizations()
        self.init_base_views()
        for k in tqdm.tqdm(range(self.n_timesteps), desc=desc):
            self.current_timestep = k + 1
            while visualizer.paused:
                visualizer.update(
                    agents=self.get_all_agents(),
                    items=self.get_all_items(),
                    pheromones=self.get_pheromones()
                )
            all_agents = self.get_random_agent_order()
            for swarm_name, agent_location in all_agents:
                if agent_location not in self.environment.swarms[swarm_name]:
                    continue
                agent = self.environment.swarms[swarm_name][agent_location]
                self.agent_turn(agent)
                self.handle_agent_post_action_updates(agent)
            self.regrow_item_energy_levels()
            self.environment.respawn_items()
            if self.environment.pheromones is not None:
                self.decay_pheromones()
            self.handle_tick_info_for_visualizations()
            stats = self.get_basic_stats(stats)
            visualizer.update(
                agents=self.get_all_agents(),
                items=self.get_all_items(),
                pheromones=self.get_pheromones(),
                draw_pheromones_flag=False,
                stats=stats if show_graphs else None
            )
            if self.stopping_criterion == 'all_agents_dead':
                n_agents_left = sum([len(swarm) for swarm in self.environment.swarms.values()])
                if n_agents_left == 0:
                    break
    pygame.quit()


def AverageNumberOfAgents(
    simulation: Simulation,
    swarm_name: str
) -> float:
    n_agents = []
    symbol = simulation.environment.swarm_configs[swarm_name]['symbol']
    for frame in simulation.frames:
        n_agents.append((frame == symbol).sum())
    score = np.mean(n_agents)
    return score


LOSSFN_MAPPER = {
    'avg_num_agents': AverageNumberOfAgents
}


class Game:
    def __init__(
        self,
        config: str,
        game_seed: int,
        game_path: str

    ):
        self.config = config
        self.game_seed = game_seed
        self.game_path = game_path
        self.game_results = None
        self.game_times = None
    
    def play_round(
        self,
        round_seed: int,
        round_number: int
    ) -> Dict:
        config_copy = copy.deepcopy(self.config)
        environment = Environment(**config_copy['environment'], seed=round_seed)
        environment.initialize_board()
        simulation = Simulation(
            environment=environment, **config_copy['simulation'], seed=round_seed,
            pygame_params=self.config['visualization']['pygame']
        )
        if self.config['visualization']['interactive']:
            simulation.simulate_pygame(f'set {round_number}')
        else:
            simulation.simulate(f'set {round_number}')
        goal_function = LOSSFN_MAPPER[self.config['tournament']['goal_function_name']]
        swarm_names = list(self.config['environment']['swarm_configs'].keys())
        round_scores = {
            swarm_name: goal_function(simulation, swarm_name) for swarm_name in swarm_names
        }
        available_runtime = config_copy['simulation']['available_runtime_per_swarm']
        round_times = {
            swarm_name: available_runtime - simulation.remaining_available_runtimes[swarm_name] for swarm_name in swarm_names
        }
        if self.game_path is not None:
            round_path = os.path.join(self.game_path, f'set_{round_number}')
            os.makedirs(round_path)
            if self.config['visualization']['save_videos']:
                if self.config['visualization']['interactive']:
                    generate_video_from_frames(
                        frames_path='video_frames',
                        output_filename=os.path.join(round_path, 'visualization.mp4'),
                        fps=self.config['visualization']['video']['fps']
                    )
                else:
                    generate_video(
                        frames=simulation.frames,
                        **self.config['visualization']['video'],
                        outfname=os.path.join(round_path, 'visualization')
                    )
            if self.config['visualization']['save_plots']:
                plots_path = os.path.join(round_path, 'plots')
                os.makedirs(plots_path)
                save_plots(
                    simulation=simulation,
                    path=plots_path,
                    plot_categories_mapper={
                        'action_counts': lambda x: x.endswith(' count'),
                        'average_agent_attribute_values': lambda x: x.startswith('average ')
                    }
                )
            if self.config['visualization']['save_logs']:
                logfiles_path = os.path.join(round_path, 'logs')
                os.makedirs(logfiles_path)
                for swarm_name in swarm_names:
                    full_errors_log = simulation.swarm_errors_log[swarm_name]
                    error_logs_path = os.path.join(logfiles_path, f'{swarm_name}_errors.json')
                    with open(error_logs_path, 'w') as fp:
                        json.dump(full_errors_log, fp)
                    full_actions_log = simulation.swarm_actions_log[swarm_name]
                    full_logs_path = os.path.join(logfiles_path, f'{swarm_name}_actions.json')
                    with open(full_logs_path, 'w') as fp:
                        json.dump(full_actions_log, fp)
        return round_scores, round_times
    

    def play_game(
        self
    ) -> pd.DataFrame:
        random.seed(self.game_seed)
        n_rounds = self.config['tournament']['n_rounds_per_game']
        round_seeds = sample(range(n_rounds * 10), n_rounds)
        scores = []
        times = []
        for i, round_seed in enumerate(round_seeds):
            round_scores, round_times = self.play_round(
                round_seed=round_seed,
                round_number=i+1
            )
            scores.append(round_scores)
            times.append(round_times)
        self.game_results = pd.DataFrame(scores)
        self.game_times = pd.DataFrame(times)
        if self.game_path is not None:
            result_table_path = os.path.join(self.game_path, 'results.csv')
            self.game_results.to_csv(result_table_path)
            times_table_path = os.path.join(self.game_path, 'times.csv')
            self.game_times.to_csv(times_table_path)


class Tournament:
    def __init__(
        self,
        config_path: str=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deathmatch_config.json'),
        competitors_path: str=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'competitors')
    ):
        self.agent_behavior_mapper = get_competitors(
            folder=competitors_path
        )
        self.swarm_names = list(self.agent_behavior_mapper.keys())
        self.config = generate_config(
            meta_config_path=config_path,
            teamnames=self.swarm_names
        )
        self.tournament_results = None
        self.leaderboard = None
        self.cumulative_scores = {
            swarm_name: 0 for swarm_name in self.swarm_names
        }
        self.n_games_played = {
            swarm_name: 0 for swarm_name in self.swarm_names
        }
        self.num_swarms_per_faceoff = self.config['tournament']['n_swarms_per_game']
        self.n_pairings, self.n_to_leave_out = divmod(
            len(self.swarm_names), self.num_swarms_per_faceoff
        )
        self.tournament_log = {
            'summary': {
                'competitors': self.swarm_names,
                'swarms_visualization_colors_rgb': {
                    swarm_name: torgb(config['symbol']) \
                        for swarm_name, config in self.config['environment']['swarm_configs'].items()
                },
                'items_visualization_colors_rgb': {
                    item_name: torgb(config['symbol']) \
                        for item_name, config in self.config['environment']['item_configs'].items()
                },
                'item_categories': list(self.config['environment']['item_configs'].keys()),
                'goal': self.config['tournament']['goal_function_name'],
                'num_tournament_rounds': self.config['tournament']['n_tournament_rounds'],
                'num_swarms_per_game': self.config['tournament']['n_swarms_per_game'],
                'num_rounds_per_game': self.config['tournament']['n_rounds_per_game'],
                'num_timesteps_per_simulation': self.config['simulation']['n_timesteps'],
                'num_seconds_per_swarm_per_simulation': self.config['simulation']['available_runtime_per_swarm']
            },
            'results': []
        }
        

    def make_leaderboard(
        self
    ) -> None:
        """
        make leaderboard from tournament results
        """
        leaderboard = pd.DataFrame(self.tournament_log['results']).T
        n_swarms, n_rounds = leaderboard.shape
        leaderboard.columns = [f'Round {i+1} Score' for i in range(n_rounds)]
        leaderboard['Cumulative Score'] = pd.Series(self.cumulative_scores)
        leaderboard.sort_values(by='Cumulative Score', ascending=False, inplace=True)
        leaderboard['Rank'] = range(1, n_swarms + 1)
        self.leaderboard = leaderboard
        results_path = self.config['tournament']['results_path']
        print('Leaderboard:')
        print(self.leaderboard)
        if results_path is not None:
            leaderboard_path = os.path.join(results_path, 'leaderboard.csv')
            leaderboard.to_csv(leaderboard_path)
    
    def do_pairings(
        self,
        swarms_ranked: List[str]
    ) -> Tuple[List[List[str]], List[str]]:
        """ pairings of competitors for games of the round """
        to_leave_out = []
        while len(to_leave_out) < self.n_to_leave_out:
            max_rank = max([num for swarm_name, num in self.n_games_played.items() if swarm_name not in to_leave_out])
            candidates = [swarm_name for swarm_name, num in self.n_games_played.items() if num == max_rank]
            to_leave_out.append(choice(candidates))
        swarms_to_pair = [swarm_name for swarm_name in swarms_ranked if swarm_name not in to_leave_out]
        pairings = [
            swarms_to_pair[
                i*self.num_swarms_per_faceoff:(i+1)*self.num_swarms_per_faceoff
            ] for i in range(self.n_pairings)
        ]
        return pairings, to_leave_out
    
    def run_tournament(
        self
    ) -> None:
        """
        carry out the tournament
        """
        results_path = self.config['tournament']['results_path']
        if results_path is not None:
            delnodes([results_path])
            os.makedirs(results_path)
        random.seed(self.config['tournament']['tournament_seed'])
        
        n_tournament_rounds = self.config['tournament']['n_tournament_rounds']
        if n_tournament_rounds is None:
            n_tournament_rounds = math.ceil(math.log2(len(self.swarm_names)))
        current_ranking = [swarm_name for swarm_name in self.swarm_names]
        shuffle(current_ranking)

        round_seeds = sample(range(n_tournament_rounds * 10), n_tournament_rounds)

        aggregation_method = self.config['tournament']['aggregation_method']
        
        for k, round_seed in enumerate(round_seeds):
            print(f'Round {k+1}/{n_tournament_rounds}:')
            random.seed(round_seed)
            game_seeds = sample(range(self.n_pairings * 10), self.n_pairings)
            round_path = None
            if results_path is not None:
                round_path = os.path.join(results_path, f'round_{k+1}')
                os.makedirs(round_path)
            pairings, missing = self.do_pairings(swarms_ranked=current_ranking)
            round_scores = dict()
            for i, (game_seed, swarm_group) in enumerate(zip(game_seeds, pairings)):
                game_path = None
                if results_path is not None:
                    game_path = os.path.join(round_path, f'game_{i+1}_' + '_'.join(swarm_group))
                    os.makedirs(game_path)
                swarm_group_fmt = ' and '.join([f'"{sn}"' for sn in swarm_group])
                print(f'Game {i+1}/{self.n_pairings}: Face-off of swarms {swarm_group_fmt}')
                config_copy = copy.deepcopy(self.config)
                config_copy['environment']['swarm_configs'] = {
                    swarm_name: swarm_config for swarm_name, swarm_config in \
                        config_copy['environment']['swarm_configs'].items() \
                            if swarm_name in swarm_group
                }
                config_copy['environment']['agent_behavior_mapper'] = self.agent_behavior_mapper
                game = Game(config=config_copy, game_seed=game_seed, game_path=game_path)
                game.play_game()
                faceoff_scores = game.game_results.aggregate(aggregation_method, axis=0).to_dict()
                for swarm_name, score in faceoff_scores.items():
                    round_scores[swarm_name] = score
                    self.n_games_played[swarm_name] += 1
            if k == 0:
                expected_score = sum([score for score in round_scores.values()]) / len(round_scores)
                for swarm_name in missing:
                    round_scores[swarm_name] = expected_score
            else:
                for swarm_name in missing:
                    round_scores[swarm_name] = self.cumulative_scores[swarm_name] / k
            for swarm_name, score in round_scores.items():
                self.cumulative_scores[swarm_name] += score
            self.tournament_log['results'].append(round_scores)
            ranking = [
                swarm_name for swarm_name, score in sorted(
                    self.cumulative_scores.items(), key=lambda item: -item[1]
                )
            ]
        self.make_leaderboard()
        self.tournament_log['summary']['cumlative_scores'] = self.cumulative_scores
        self.tournament_log['summary']['winner'] = ranking[0]
        if results_path is not None:
            tournament_log_path = os.path.join(results_path, 'tournament_log.json')
            with open(tournament_log_path, 'w') as fp:
                json.dump(self.tournament_log, fp)
