import pdb

import numpy as np
import itertools
from typing import List, Generator
import ConfigSpace.c_util
from openbox.utils.config_space import Configuration, ConfigurationSpace
from ConfigSpace.util import impute_inactive_values, deactivate_inactive_hyperparameters
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant, \
    OrdinalHyperparameter, NumericalHyperparameter


def convert_configurations_to_array(configs: List[Configuration]) -> np.ndarray:
    """Impute inactive hyperparameters in configurations with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configs : List[Configuration]
        List of configuration objects.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    configs_array = np.array([config.get_array() for config in configs],
                             dtype=np.float64)
    configuration_space = configs[0].configuration_space
    return impute_default_values(configuration_space, configs_array)


def impute_default_values(
        configuration_space: ConfigurationSpace,
        configs_array: np.ndarray) -> np.ndarray:
    """Impute inactive hyperparameters in configuration array with their default.

    Necessary to apply an EPM to the data.

    Parameters
    ----------
    configuration_space : ConfigurationSpace
    
    configs_array : np.ndarray
        Array of configurations.

    Returns
    -------
    np.ndarray
        Array with configuration hyperparameters. Inactive values are imputed
        with their default value.
    """
    for hp in configuration_space.get_hyperparameters():
        default = hp.normalized_default_value
        idx = configuration_space.get_idx_by_hyperparameter_name(hp.name)
        nonfinite_mask = ~np.isfinite(configs_array[:, idx])
        configs_array[nonfinite_mask, idx] = default

    return configs_array


# return neighbors without redundant indexes
def get_one_exchange_neighbourhood(
        configuration: Configuration,
        seed: int,
        num_neighbors: int = 4,
        stdev: float = 0.2):
    random = np.random.RandomState(seed)
    array = configuration.get_array()
    configuration_space = configuration.configuration_space  # type: ConfigurationSpace
    hyperparameters_usable = list(configuration.keys())

    while len(hyperparameters_usable) > 0:
        hp_name = random.choice(hyperparameters_usable)
        hyperparameters_usable.remove(hp_name)

        hp_name = str(hp_name)
        hp_idx = configuration_space.get_idx_by_hyperparameter_name(hp_name)

        new_array = array.copy()
        new_array[hp_idx] = 1 - new_array[hp_idx]

        # index turn on, potentially violation of conditions
        if new_array[hp_idx] == 1.:
            parents_idx = [
                configuration_space.get_idx_by_hyperparameter_name(p.name)
                for p in set(configuration_space.get_parents_of(hp_name))
            ]
            if sum(new_array[parents_idx]) > 0:
                continue

            children_idx = [
                configuration_space.get_idx_by_hyperparameter_name(c.name)
                for c in set(configuration_space.get_children_of(hp_name))
            ]
            new_array[children_idx] = np.nan
            new_configuration = Configuration(configuration_space, vector=new_array)
            new_configuration.is_valid_configuration()
            yield new_configuration

        # index turn off, potentially new activated indexes
        else:
            new_hps = set()
            children = set(configuration_space.get_children_of(hp_name))
            for child in children:
                all_parents_idx = [
                    configuration_space.get_idx_by_hyperparameter_name(p.name)
                    for p in set(configuration_space.get_parents_of(child.name))
                ]
                if len(np.argwhere(new_array[all_parents_idx] == 1)) == 0:
                    new_hps.add(child.name)

            new_hps = list(new_hps)
            if len(new_hps) > 0:
                new_children_idx = [
                    configuration_space.get_idx_by_hyperparameter_name(c)
                    for c in new_hps
                ]
                all_possible_values = list(itertools.product([0., 1.], repeat=len(new_hps)))
                for values in all_possible_values:
                    _new_array = new_array.copy()
                    _new_array[new_children_idx] = values
                    try:
                        new_configuration = Configuration(configuration_space, vector=_new_array)
                        new_configuration.is_valid_configuration()
                        yield new_configuration
                    except ValueError as e:
                        pass

            else:
                new_configuration = Configuration(configuration_space, vector=new_array)
                new_configuration.is_valid_configuration()
                yield new_configuration


def drop_one_index(configuration: Configuration, seed=1):
    np.random.seed(seed)

    array = configuration.get_array()
    configuration_space = configuration.configuration_space
    hyperparameters_usable = [k for k in configuration.keys() if configuration[k] == 'on']

    hp_name = np.random.choice(hyperparameters_usable)
    hyperparameters_usable.remove(hp_name)
    # print('choose [{}]'.format(hp_name))

    hp_name = str(hp_name)
    hp_idx = configuration_space.get_idx_by_hyperparameter_name(hp_name)

    new_array = array.copy()
    new_array[hp_idx] = 0

    new_hps_tmp = set()
    children = set(configuration_space.get_children_of(hp_name))
    # print('all children: {}'.format(set([c.name for c in children])))
    for child in children:
        all_parents_idx = [
            configuration_space.get_idx_by_hyperparameter_name(p.name)
            for p in set(configuration_space.get_parents_of(child.name))
        ]
        # print('child [{}] has parents: {}, with idx {}'.format(
        #     child.name,
        #     set([p.name for p in configuration_space.get_parents_of(child.name)]),
        #     all_parents_idx
        # ))
        if len(np.argwhere(new_array[all_parents_idx] == 1)) == 0:
            new_hps_tmp.add(child)
            # print(child.name + ' is activated')

    new_hps_tmp = list(new_hps_tmp)
    new_hps = list()
    for new_hp in new_hps_tmp:
        redundant = False
        for other_hp in new_hps_tmp:
            if other_hp == new_hp:
                continue
            if other_hp in configuration_space.get_parents_of(new_hp.name):
                redundant = True
                break
        if not redundant:
            new_hps.append(new_hp)

    for new_hp in new_hps:
        new_hp_idx = configuration_space.get_idx_by_hyperparameter_name(new_hp.name)

        # optional: add as many index as possible
        new_array[new_hp_idx] = 1.
        # new_array[new_hp_idx] = 0.

    new_configuration = Configuration(configuration_space, vector=new_array)
    new_configuration.is_valid_configuration()
    return new_configuration


def add_one_index(configuration: Configuration, hp):
    # TODO:
    raise NotImplementedError
