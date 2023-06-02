import pdb
import time
import json
import collections
import numpy as np
from MFIX.index import Index
from MFIX.utils.constants import MAXINT, SUCCESS
from MFIX.utils.config_space import Configuration, ConfigurationSpace
from MFIX.utils.transform import get_transform_function
from MFIX.utils.logging_utils import get_logger

Observation = collections.namedtuple(
    'Observation', [
        'config',  # list or configuration
        'perf',
        'constraint_perf',  # constraint_perf = storage_cost - storage_budget
        'trial_state',
        'elapsed_time',
        'index_modify_time',
        'timestamp'
    ]
)


class HistoryContainer:
    def __init__(
            self,
            task_id,
            method,
            index_storage_budget,
            fidelity,
            timeout
    ):
        self.task_id = task_id
        self.method = method
        self.fidelity = fidelity
        self.index_storage_budget = index_storage_budget
        self.timeout = timeout

        self.logger = get_logger(self.__class__.__name__)

        self.config_counter = 0
        self.observation_counter = 0
        self.data = collections.OrderedDict()
        self.incumbents = list()
        self.incumbent_value = None

        self.configs = list()
        self.perfs = list()
        self.constraint_perfs = list()
        self.trial_states = list()
        self.elapsed_times = list()
        self.index_modify_times = list()
        self.timestamps = list()
        self.history = list()

        self.global_start_time = time.time()
        self.scale_perc = 5
        self.perc = None
        self.min_y = None
        self.max_y = MAXINT

    def update_observation(self, observation: Observation):
        self.configs.append(observation.config)
        self.perfs.append(observation.perf)
        self.constraint_perfs.append(observation.constraint_perf)
        self.trial_states.append(observation.trial_state)
        self.elapsed_times.append(observation.elapsed_time)
        self.index_modify_times.append(observation.index_modify_time)
        self.timestamps.append(observation.timestamp)
        self.observation_counter += 1

        self.update_history()

        if observation.trial_state == SUCCESS and observation.constraint_perf <= 0:
            if self.incumbent_value is None:
                self.incumbent_value = observation.perf
                self.incumbents.append((observation.config, observation.perf))
            else:
                if observation.perf < self.incumbent_value:
                    self.incumbents.clear()
                if observation.perf <= self.incumbent_value:
                    self.incumbent_value = observation.perf
                    self.incumbents.append((observation.config, observation.perf))

    def update_history(self):
        self.history.append({
                'config': [str(index) for index in self.configs[-1]],
                'perf': self.perfs[-1],
                'constraint_perf': self.constraint_perfs[-1],
                'trial_state': self.trial_states[-1],
                'elapsed_time': self.elapsed_times[-1],
                'index_modify_time': self.index_modify_times[-1],
                'timestamp': self.timestamps[-1]
            })

    def get_perf(self, config: Configuration):
        return self.data[config]

    def get_all_perfs(self):
        return list(self.data.values())

    def get_all_configs(self):
        return list(self.data.keys())

    def empty(self):
        return self.config_counter == 0

    def get_incumbents(self):
        return self.incumbents

    def save_history_to_json(self, filename):
        with open(filename, "w") as fp:
            json.dump({
                'task_id': self.task_id,
                'method': self.method,
                'fidelity': self.fidelity,
                'index_storage_budget': self.index_storage_budget,
                'global_start_time': self.global_start_time,
                'timeout': self.timeout,
                'history': self.history,
                'best': [str(index) for index in self.incumbents[0][0]] if len(self.incumbents) else None
            }, fp, indent=4)

    def load_history_from_json(self, filename, config_space=None, columns=None):
        try:
            with open(filename) as fp:
                data = json.load(fp)
        except Exception as e:
            self.logger.warning(
                'Encountered exception %s while reading run history from %s. '
                'Not adding any runs!', e, filename,
            )
            return

        self.fidelity = data['fidelity']
        self.index_storage_budget = data['index_storage_budget']
        self.timeout = data['timeout']

        history = data["history"]
        for tmp in history:
            config = self.load_index_config_from_string(tmp['config'], columns)

            observation = Observation(
                config=config,
                perf=tmp['perf'],
                constraint_perf=tmp['constraint_perf'],
                trial_state=tmp['trial_state'],
                elapsed_time=tmp['elapsed_time'],
                index_modify_time=tmp['index_modify_time'],
                timestamp=tmp['timestamp']
            )
            self.update_observation(observation)

        self.logger.info(
            f"load run history from {filename}."
            f"\n  load {self.observation_counter} observations."
            f"\n  index_storage_budget: {self.index_storage_budget}"
            f"\n  fidelity: {self.fidelity}"
            f"\n  timeout: {self.timeout}"
        )

    @staticmethod
    def load_index_config_from_string(index_config_string, columns):
        config = []
        for index_string in index_config_string:
            column_strings = index_string.split('I(')[-1].split(')')[0].split(',')
            columns_list = []
            for column_string in column_strings:
                for _ in columns:
                    if str(_) == column_string:
                        columns_list.append(_)
            config.append(Index(columns_list))

        return config


class BOHistoryContainer(HistoryContainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.successful_perfs = list()
        self.failed_index = list()
        self.transform_perf_index = list()

    def update_observation(self, observation):
        super().update_observation(observation)

        transform_perf = False
        failed = False
        if observation.trial_state == SUCCESS and observation.perf < MAXINT:
            self.successful_perfs.append(observation.perf)
            if observation.constraint_perf < 0:
                feasible = True
            else:
                feasible = False
                transform_perf = True

            if feasible:
                self.add(observation.config, observation.perf)
            else:
                self.add(observation.config, MAXINT)
        else:
            failed = True
            transform_perf = True

        cur_index = self.observation_counter - 1
        if transform_perf:
            self.transform_perf_index.append(cur_index)
        if failed:
            self.failed_index.append(cur_index)

    def add(self, config: Configuration, perf):
        if config in self.data.keys():
            # self.logger.warning('Repeated configuration detected!')
            return

        self.data[config] = perf
        self.config_counter += 1

        self.perc = np.percentile(self.successful_perfs, self.scale_perc, axis=0)
        self.min_y = np.min(self.successful_perfs, axis=0).tolist()
        self.max_y = np.max(self.successful_perfs, axis=0).tolist()

    def get_transformed_perfs(self, transform=None):
        # set perf of failed trials to current max
        transformed_perfs = self.perfs.copy()
        for i in self.transform_perf_index:
            transformed_perfs[i] = self.timeout

        transformed_perfs = np.array(transformed_perfs, dtype=np.float64)
        transformed_perfs = get_transform_function(transform)(transformed_perfs)
        return transformed_perfs

    def update_history(self):
        self.history.append({
                'config': [k for k, v in self.configs[-1].get_dictionary().items() if v == 'on'],
                'perf': self.perfs[-1],
                'constraint_perf': self.constraint_perfs[-1],
                'trial_state': self.trial_states[-1],
                'elapsed_time': self.elapsed_times[-1],
                'index_modify_time': self.index_modify_times[-1],
                'timestamp': self.timestamps[-1]
            })

    def load_history_from_json(self, filename, config_space=None, columns=None):
        if config_space is None:
            super().load_history_from_json(filename, config_space, columns)

        try:
            with open(filename) as fp:
                data = json.load(fp)
        except Exception as e:
            self.logger.warning(
                'Encountered exception %s while reading run history from %s. '
                'Not adding any runs!', e, filename,
            )
            return

        self.fidelity = data['fidelity']
        self.index_storage_budget = data['index_storage_budget']
        self.timeout = data['timeout']

        assert config_space and isinstance(config_space, ConfigurationSpace)

        history = data["history"]
        for tmp in history:
            config = tmp['config']
            config_dict = {}
            for hp in config:
                if hp not in config_space.get_hyperparameter_names():
                    config.remove(hp)
                    self.logger.debug(f"{hp} removed")
                else:
                    config_dict[hp] = 'on'
            for hp in config_space.get_hyperparameter_names():
                if hp not in config_dict:
                    config_dict[hp] = 'off'

            config = Configuration(config_space, config_dict, allow_inactive_with_values=True)
            active_hps = config_space.get_active_hyperparameters(config)
            config_dict_valid = {}
            for hp in config_dict:
                if hp in active_hps:
                    config_dict_valid[hp] = config_dict[hp]

            config = Configuration(config_space, config_dict_valid)
            config.is_valid_configuration()

            observation = Observation(
                config=config,
                perf=tmp['perf'],
                constraint_perf=tmp['constraint_perf'],
                trial_state=tmp['trial_state'],
                elapsed_time=tmp['elapsed_time'],
                index_modify_time=tmp['index_modify_time'],
                timestamp=tmp['timestamp']
            )
            self.update_observation(observation)

        self.logger.info(
            f"load run history from {filename}."
            f"\n  load {self.observation_counter} observations (successful: {self.config_counter})."
            f"\n  index_storage_budget: {self.index_storage_budget}"
            f"\n  fidelity: {self.fidelity}"
            f"\n  timeout: {self.timeout}"
        )
