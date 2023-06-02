import copy
import os.path
import pdb
import time
import json
import numpy as np
from tqdm import tqdm
from MFIX.algorithms.base import IndexSelectionAlgorithm
from MFIX.algorithms.model import RandomForestWithInstances, IndexStorageModel, RGPE
from MFIX.algorithms.acquisition_function.acquisition import EIC, LCBC
from MFIX.algorithms.acq_optimizer.optmization import InterleavedLocalAndRandomSearch
from MFIX.cost_evaluation import LowFidelityEvaluation
from MFIX.utils.index_utils import get_utilized_indexes
from MFIX.utils.history_utils import Observation
from MFIX.utils.logging_utils import get_logger, setup_logger
from MFIX.utils.constants import MAXINT, SUCCESS, FAILED, TIMEOUT
from MFIX.utils.sampler import max_min_distance
from MFIX.candidate_generation import candidates_per_query, syntactically_relevant_indexes_per_clause
from MFIX.utils.config_space import ConfigurationSpace, \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter, \
    AndConjunction, EqualsCondition, drop_one_index, add_one_index, convert_configurations_to_array


class BO(IndexSelectionAlgorithm):
    def __init__(
            self,
            surrogate_type='prf',
            constraint_surrogate_type='linear',
            acq_type='eic',
            acq_optimizer_type='local_random',
            max_runs=200,
            max_runtime_minutes=60,
            init_runs=10,
            init_strategy='random_explore_first',
            rand_prob=0.2,
            random_state=1,
            max_index_width=2,
            utilized_index_only=True,
            add_merged_index=True,
            include_dta=True,
            use_dta=False,
            **kwargs
    ):
        self.max_index_width = int(max_index_width)
        self.utilized_index_only = eval(utilized_index_only) if type(utilized_index_only) is str else utilized_index_only
        self.add_merged_index = eval(add_merged_index) if type(add_merged_index) is str else add_merged_index
        self.include_dta = eval(include_dta) if type(include_dta) is str else include_dta
        self.use_dta = eval(use_dta) if type(use_dta) is str else use_dta

        self.max_runs = int(max_runs)
        self.max_time = int(max_runtime_minutes) * 60
        self.init_num = int(init_runs)
        self.init_strategy = init_strategy
        self.surrogate_type = surrogate_type
        self.constraint_surrogate_type = constraint_surrogate_type
        self.acq_type = acq_type
        self.acq_optimizer_type = acq_optimizer_type
        self.rand_prob = float(rand_prob)
        self.random_state = int(random_state)
        self.rng = np.random.RandomState(int(random_state))

        # BO basic ingredients
        self.surrogate_model = None
        self.constraint_surrogate_model = None
        self.acquisition_function = None
        self.optimizer = None

        self.candidates = None
        self.config_space = None
        self.types = None
        self.bounds = None
        self.initial_configurations = None

        self.iteration_id = 0
        super().__init__(**kwargs)

    def _calculate_best_index_configuration(self, workload):
        self.workload = workload

        self.setup_config_space()
        self.setup_bo_basics()
        self.initial_configurations = self.create_initial_design(self.init_strategy, init_num=self.init_num)
        self.logger.info(f'create {len(self.initial_configurations)} initial design.')

        global_start_time = time.time()

        for _ in tqdm(range(self.max_runs)):
            self.iterate()
            consumed_time = time.time() - global_start_time
            if consumed_time > self.max_time:
                self.logger.info(
                    f"Stopping after {_ + 1} iterations because of timing constraints."
                )
                break

        incumbent = [k for k, v in self.history_container.get_incumbents()[0][0].get_dictionary().items() if v == 'on']
        best_perf = self.history_container.get_incumbents()[0][1]
        best_index_configuration = []
        for index in self.candidates:
            if str(index) in incumbent:
                best_index_configuration.append(index)

        return best_index_configuration, best_perf

    def iterate(self):
        self.iteration_id += 1
        start_time = time.time()

        config = self.get_suggestion()
        indexes = {
            candidate for candidate in self.candidates
            if str(candidate) in config.get_dictionary() and config[str(candidate)] == "on"
        }

        try:
            cost, _, index_modify_time = self.cost_evaluation.calculate_cost(self.workload, indexes)
            trial_state = SUCCESS
        except Exception as e:
            self.logger.warning(e)
            trial_state = FAILED
            cost, index_modify_time = -1, -1

        storage_cost = self.constraint_surrogate_model.get_cost(config)
        # storage_cost = self.cost_evaluation.get_current_size_total()

        elapsed_time = time.time() - start_time

        self.history_container.update_observation(Observation(
            config=config,
            perf=cost,
            constraint_perf=storage_cost - self.index_storage_budget,
            trial_state=trial_state,
            elapsed_time=elapsed_time,
            index_modify_time=index_modify_time,
            timestamp=time.time()
        ))
        self.history_container.save_history_to_json(self.history_file)

        if trial_state == FAILED:
            self.logger.info(f'Iteration {self.iteration_id} FAILED!')
        elif trial_state == TIMEOUT:
            self.logger.info(f'Iteration {self.iteration_id} TIMEOUT!')
        else:
            if self.iteration_id == 1:
                self.initial_cost = cost
                self.logger.info(
                    f'Iteration {self.iteration_id}, '
                    f'Initial cost: {cost:.2f}, '
                    f'size: {storage_cost:.2f}MB.'
                )
            else:
                self.logger.info(
                    f'Iteration {self.iteration_id}, '
                    f'cost: {cost:.2f} (imp. {(1 - cost / self.initial_cost) * 100:.2f}%), '
                    f'size: {storage_cost:.2f}MB.'
                    f'\t(current best imp. '
                    f'{(1 - self.history_container.get_incumbents()[0][1] / self.initial_cost) * 100:.2f}%)'
                )

    def setup_config_space(self):
        self.candidates = self._generate_candidates(
            utilized_index_only=self.utilized_index_only,
            add_merged_index=self.add_merged_index,
            include_dta=self.include_dta,
            use_dta=self.use_dta
        )

        self.config_space = ConfigurationSpace()
        for c in self.candidates:
            self.config_space.add_hyperparameter(
                CategoricalHyperparameter(str(c), ["off", "on"])
            )

        parent_candidates = dict()
        for c1 in self.candidates:
            for c2 in self.candidates:
                if c1 != c2 and c1.subsumes(c2):
                    if c2 not in parent_candidates.keys():
                        parent_candidates[c2] = set()
                    parent_candidates[c2].add(c1)

        for c, ps in parent_candidates.items():
            ps = sorted(ps)
            condition = EqualsCondition(
                self.config_space.get_hyperparameter(str(c)),
                self.config_space.get_hyperparameter(str(ps[0])), "off"
            )
            if len(ps) > 1:
                for p in ps[1:]:
                    condition = AndConjunction(
                        EqualsCondition(
                            self.config_space.get_hyperparameter(str(c)),
                            self.config_space.get_hyperparameter(str(p)), "off"
                        ),
                        condition
                    )
            self.config_space.add_condition(condition)
        self.config_space.seed(self.rng.randint(MAXINT))

        self.types = np.zeros(len(self.config_space.get_hyperparameters()), dtype=np.uint)
        self.bounds = [(np.nan, np.nan)] * self.types.shape[0]
        for i, hp in enumerate(self.config_space.get_hyperparameters()):
            if isinstance(hp, CategoricalHyperparameter):
                n_cats = len(hp.choices)
                self.types[i] = n_cats
                self.bounds[i] = (int(n_cats), np.nan)
            elif isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                self.bounds[i] = (0, 1.0)

    def setup_bo_basics(self, source_hpo_data=None):
        self.surrogate_model = self.build_surrogate(surrogate_type=self.surrogate_type, source_hpo_data=source_hpo_data)
        self.constraint_surrogate_model = self.build_surrogate(surrogate_type=self.constraint_surrogate_type)
        self.acquisition_function = self.build_acq_func(acq_type=self.acq_type)
        self.optimizer = self.build_optimizer(acq_optimizer_type=self.acq_optimizer_type)
        self.logger.info(
            f'Build BO basic ingredients:\n\n'
            f'\t[Surrogate Model] {self.surrogate_model.__class__.__name__}\n'
            f'\t[Constraint Surrogate Model] {self.constraint_surrogate_model.__class__.__name__}\n'
            f'\t[Acquisition Function] {self.acquisition_function.__class__.__name__}\n'
            f'\t[Optimizer] {self.optimizer.__class__.__name__}\n'
        )

    def build_surrogate(self, surrogate_type, source_hpo_data=None):
        if surrogate_type.startswith('rgpe'):
            seed = self.rng.randint(MAXINT)
            inner_surrogate_type = surrogate_type.split('_')[-1]
            return RGPE(
                self.config_space, source_hpo_data, seed,
                surrogate_type=inner_surrogate_type, num_src_hpo_trial=-1
            )
        elif surrogate_type == 'prf':
            seed = self.rng.randint(MAXINT)
            return RandomForestWithInstances(
                types=self.types,
                bounds=self.bounds,
                seed=seed
            )
        elif surrogate_type == 'linear':
            model = IndexStorageModel(
                candidates=self.candidates,
                index_storage_budget=self.index_storage_budget,
                config_space=self.config_space
            )
            return model
        else:
            raise NotImplementedError

    def build_acq_func(self, acq_type=None):
        if acq_type is None:
            acq_type = self.acq_type

        if acq_type == 'eic':
            return EIC(model=self.surrogate_model,
                       constraint_models=[self.constraint_surrogate_model, ])
        elif acq_type == 'lcbc':
            return LCBC(model=self.surrogate_model,
                        constraint_models=[self.constraint_surrogate_model, ])
        else:
            raise NotImplementedError

    def build_optimizer(self, acq_optimizer_type=None):
        if acq_optimizer_type is None:
            acq_optimizer_type = self.acq_optimizer_type

        if acq_optimizer_type == 'local_random':
            return InterleavedLocalAndRandomSearch(acquisition_function=self.acquisition_function,
                                                   config_space=self.config_space,
                                                   cost_model=self.constraint_surrogate_model,
                                                   rng=self.rng)
        else:
            return NotImplementedError

    def create_initial_design(self, init_strategy='default', init_num=10, excluded_configs=None,
                              include_default=True, default_config=None):
        if not default_config:
            default_config = self.config_space.get_default_configuration()

        num_random_config = init_num - 1
        if init_strategy == 'random':
            initial_configs = self.sample_feasible_random_configs(init_num, excluded_configs)
            return initial_configs
        elif init_strategy == 'default':
            initial_configs = [default_config] + self.sample_feasible_random_configs(num_random_config,
                                                                                     excluded_configs)
            return initial_configs
        elif init_strategy == 'random_explore_first':
            candidate_configs = self.sample_feasible_random_configs(init_num * 10, excluded_configs)
            if include_default:
                return max_min_distance(default_config, candidate_configs, num_random_config)
            else:
                return max_min_distance(default_config, candidate_configs, init_num)[1:]
        else:
            raise ValueError('Unknown initial design strategy: %s.' % init_strategy)

    def sample_feasible_random_configs(self, num_configs=1, excluded_configs=None):
        if excluded_configs is None:
            excluded_configs = set()

        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            sample_cnt += 1
            config = self.config_space.sample_configuration()

            if config not in configs and config not in excluded_configs:
                if self.constraint_surrogate_model.get_cost(config) < self.index_storage_budget:
                    configs.append(config)
                    sample_cnt = 0
                    continue

            if sample_cnt >= max_sample_cnt:
                # self.logger.debug(
                #     f'Cannot find a feasible configuration in {max_sample_cnt} steps, random drop.'
                # )

                while self.constraint_surrogate_model.get_cost(config) > self.index_storage_budget:
                    config = drop_one_index(config)

                config.is_valid_configuration()
                if config not in configs and config not in excluded_configs:
                    configs.append(config)
                sample_cnt = 0

        return configs

    def get_suggestion(self):
        num_config_evaluated = len(self.history_container.configs)
        num_config_successful = len(self.history_container.successful_perfs)

        if num_config_evaluated < self.init_num:
            self.logger.info(
                f'Initial config, predicated storage size '
                f'{self.constraint_surrogate_model.get_cost(self.initial_configurations[num_config_evaluated])}MB.'
            )
            return self.initial_configurations[num_config_evaluated]

        if self.rng.random() < self.rand_prob:
            self.logger.info('Sample random config. rand_prob=%.2f.' % self.rand_prob)
            return self.sample_feasible_random_configs(1, self.history_container.configs)[0]

            # chosen_index = self.rng.choice(self.never_chosen_candidates, 1)
            # self.never_chosen_candidates.remove(chosen_index)
            # self.logger.info(
            #     f'Sample random config with never chosen index {chosen_index}. rand_prob={self.rand_prob:.2f}'
            # )
            # return self.sample_feasible_random_configs(1, self.history_container.configs, input_index=chosen_index)[0]

        X = convert_configurations_to_array(self.history_container.configs)
        Y = self.history_container.get_transformed_perfs()

        self.surrogate_model.train(X, Y)

        incumbent_value = self.history_container.get_incumbents()[0][1]
        self.acquisition_function.update(
            model=self.surrogate_model,
            constraint_models=[self.constraint_surrogate_model, ],
            eta=incumbent_value,
            num_data=num_config_evaluated
        )

        challengers = self.optimizer.maximize(runhistory=self.history_container, num_points=5000)

        is_repeated_config = True
        repeated_time = 0
        cur_config = None
        while is_repeated_config:
            cur_config = challengers.challengers[repeated_time]
            if cur_config in self.history_container.configs:
                repeated_time += 1
            else:
                is_repeated_config = False

        return cur_config

    def _generate_candidates(self, utilized_index_only=True, add_merged_index=False, include_dta=True, use_dta=False):
        if use_dta:
            self.logger.info('use DTA candidates.')
            low_fidelity_cost_evaluation = LowFidelityEvaluation(self.db_conn)
            candidates = super()._generate_candidates(
                utilized_index_only=True,
                add_merged_index=True,
                filtered=True,
                cost_evaluation=low_fidelity_cost_evaluation
            )
            candidates = sorted(candidates)

            # generate candidates sizes before filtering.
            candidates = self._generate_candidate_sizes(candidates, filename=f"{self.benchmark}_index_size.json")
            return candidates

        self.logger.info('generate index candidates.')

        syntactical_candidates_list = candidates_per_query(
            self.workload,
            self.max_index_width,
            candidate_generator=syntactically_relevant_indexes_per_clause,
            db_conn=self.db_conn,
            workload_pickle_file=self.workload_pickle_file
        )

        syntactical_candidates_set = set([c for cs in syntactical_candidates_list for c in cs])
        self.logger.debug(f'synthetically relevant candidates per clause: {len(syntactical_candidates_set)}.')

        if utilized_index_only:
            low_fidelity_cost_evaluation = LowFidelityEvaluation(self.db_conn)
            utilized_candidates, _ = get_utilized_indexes(
                self.workload, syntactical_candidates_list, low_fidelity_cost_evaluation
            )
            low_fidelity_cost_evaluation.complete_cost_evaluation()

            self.logger.info(f'utilized indexes: {len(utilized_candidates)}.')
            self.logger.debug(
                f'utilized indexes details:\n' + '\n'.join([str(_) for _ in utilized_candidates])
            )
            candidates = utilized_candidates
        else:
            candidates = syntactical_candidates_set

        if add_merged_index:
            low_fidelity_cost_evaluation = LowFidelityEvaluation(self.db_conn)
            merged_indexes = self._add_merged_indexes(candidates)
            low_fidelity_cost_evaluation.complete_cost_evaluation()

            self.logger.debug(f'add merged indexes: {len(merged_indexes)}.')
            self.logger.debug(
                f'merged indexes details:\n' + '\n'.join([str(_) for _ in merged_indexes])
            )

        if include_dta:
            add_dta_num = 0
            self.logger.debug('(include_DTA=True) generate DTA index candidates.')
            low_fidelity_cost_evaluation = LowFidelityEvaluation(self.db_conn)
            dta_candidates = super()._generate_candidates(filtered=False, cost_evaluation=low_fidelity_cost_evaluation)
            low_fidelity_cost_evaluation.complete_cost_evaluation()

            for dta_index in dta_candidates:
                dta_columns = set(dta_index.columns)
                exist = False
                for index in candidates:
                    columns = set(index.columns)
                    if dta_columns == columns:  # index with same columns but different order
                        exist = True
                        break
                if not exist:
                    candidates.add(dta_index)
                    add_dta_num += 1
                    self.logger.debug(f'add DTA candidate {dta_index}.')

            self.logger.info(f'add DTA indexes: {add_dta_num}.')

        # generate candidates sizes before filtering.
        candidates = self._generate_candidate_sizes(candidates, filename=f"{self.benchmark}_index_size.json")

        filtered_candidates = self._filter_candidates(candidates)
        self.logger.info(f'number of candidate indexes: {len(filtered_candidates)}.')

        filtered_candidates = sorted(filtered_candidates)
        return filtered_candidates

    def setup_logger(self, log_file=None):
        if log_file is None:
            log_file = os.path.join(self.logging_dir, f'{self.method}_{self.random_state}.log')

        setup_logger(output_file=log_file)
        self.logger.info(f'save logs to {log_file}')

    def generate_history(self, history_dir=None):
        if history_dir is None:
            history_dir = self.history_dir
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        history_file = os.path.join(history_dir, f"{self.task_id}_{self.method}_{self.random_state}.json")
        return history_file
