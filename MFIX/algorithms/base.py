import os
import pdb
import json
import time
import itertools
from MFIX.cost_evaluation import LowFidelityEvaluation, HighFidelityEvaluation
from MFIX.index import Index, index_merge
from MFIX.candidate_generation import candidates_per_query, syntactically_relevant_indexes
from MFIX.utils.history_utils import Observation
from MFIX.utils.constants import SUCCESS, FAILED
from MFIX.utils.index_utils import get_utilized_indexes, indexes_by_table
from MFIX.utils.history_utils import HistoryContainer, BOHistoryContainer
from MFIX.utils.logging_utils import get_logger, setup_logger
from colorama import Fore, Back, Style


class IndexSelectionAlgorithm:
    def __init__(
            self,
            task_id,
            benchmark,
            db_conn,
            index_storage_budget=0,
            max_index_width=2,
            fidelity=0,
            timeout=10,
            logging_dir='logs',
            history_dir='history/',
            workload_pickle_file=None,
            tables=None,
            columns=None,
            **kwargs
    ):
        self.method = self.__class__.__name__
        self.task_id = task_id
        self.benchmark = benchmark
        self.index_storage_budget = int(index_storage_budget)
        self.max_index_width = int(max_index_width)
        self.fidelity = int(fidelity)
        self.timeout = int(timeout)

        self.logging_dir = logging_dir
        self.logger = get_logger(self.__class__.__name__)
        self.history_dir = history_dir
        self.history_file = self.generate_history(history_dir)
        self.history_container = None
        self.workload_pickle_file = workload_pickle_file

        self.db_conn = db_conn
        self.tables = tables
        self.columns = columns
        self.workload = None
        self.cost_evaluation = None

        self.initial_cost = None
        self.best_cost = None
        self.best_configuration = None
        self.total_elapsed_time = None

    def calculate_best_index_configuration(self, workload):
        self.setup_logger()
        self.logger.info("calculating best indexes...")

        if self.method in ['NoIndex', 'BO']:
            self.history_container = self.setup_history_container()

            if self.fidelity == 0:
                self.logger.info(f"evaluate with WHAT-IF CALLS.")
                self.cost_evaluation = LowFidelityEvaluation(self.db_conn)
            elif self.fidelity == 1:
                self.cost_evaluation = HighFidelityEvaluation(self.db_conn, timeout=int(self.timeout))
                self.logger.info(f"evaluate with QUERY EXECUTION.")

        self.best_configuration, self.best_cost = self._calculate_best_index_configuration(workload)
        self.total_elapsed_time = time.time() - self.history_container.global_start_time
        self.cost_evaluation.complete_cost_evaluation()

    def _calculate_best_index_configuration(self, workload):
        raise NotImplementedError

    def load_best_index_configuration(self, history_file=None):
        self.logger.info("loading best indexes...")
        self.history_container = self.load_history_container(history_file=history_file)
        self.best_configuration, self.best_cost = self.history_container.get_incumbents()[0]

    def evaluate_best_index_configuration(self, workload, index_config=None, fidelity=1, timeout=600):
        self.logger.info(f"evaluating best indexes (fidelity={fidelity})...")

        if index_config is None:
            index_config = self.best_configuration

        if fidelity == 1:
            evaluation = HighFidelityEvaluation(self.db_conn, timeout=timeout)
            total_cost, cost_per_query,  _ = evaluation.calculate_cost(workload, index_config, verbose=True)
            evaluation.complete_cost_evaluation()
        else:
            evaluation = LowFidelityEvaluation(self.db_conn)
            total_cost, cost_per_query, _ = evaluation.calculate_cost(workload, index_config, verbose=True)
            evaluation.complete_cost_evaluation()

        self.logger.info(Fore.BLUE + f"total cost: {total_cost}" + Style.RESET_ALL)
        return total_cost, cost_per_query

    '''
    index-related
    '''

    def prepare_index_and_evaluate_cost(self, workload, indexes):
        return self._prepare_index_and_evaluate_cost(workload, indexes)

    def generate_candidates(self, utilized_index_only=True, add_merged_index=True, filtered=True):
        return self._generate_candidates(utilized_index_only, add_merged_index, filtered)

    def _prepare_index_and_evaluate_cost(self, workload, indexes, verbose=False):
        start_time = time.time()

        try:
            cost, _, index_modify_cost = self.cost_evaluation.calculate_cost(
                workload, indexes, store_size=True, verbose=verbose
            )
            trial_state = SUCCESS
        except Exception as e:
            self.logger.warning(f'FAILED: {e}')
            cost, index_modify_cost = -1, -1
            trial_state = FAILED

        elapsed_time = time.time() - start_time
        storage_cost = self.cost_evaluation.get_current_size_total()

        observation = Observation(
            config=indexes,
            perf=cost,
            constraint_perf=storage_cost - self.index_storage_budget,
            trial_state=trial_state,
            elapsed_time=elapsed_time,
            index_modify_time=index_modify_cost,
            timestamp=time.time()
        )
        self.history_container.update_observation(observation)
        self.history_container.save_history_to_json(self.history_file)

        return cost

    def _generate_candidates(self, utilized_index_only=True, add_merged_index=True, filtered=True, cost_evaluation=None):
        if cost_evaluation is None:
            cost_evaluation = self.cost_evaluation

        syntactical_candidates_list = candidates_per_query(
            self.workload,
            self.max_index_width,
            candidate_generator=syntactically_relevant_indexes
        )

        syntactical_candidates_set = set([c for cs in syntactical_candidates_list for c in cs])
        self.logger.debug(
            f'synthetically relevant indexes (width <= {self.max_index_width}): {len(syntactical_candidates_set)}.'
        )

        if utilized_index_only:
            utilized_candidates, _ = get_utilized_indexes(self.workload, syntactical_candidates_list, cost_evaluation)
            self.logger.debug(f'utilized indexes: {len(utilized_candidates)}.')
            self.logger.debug(
                f'utilized indexes details:\n' + '\n'.join([str(_) for _ in utilized_candidates])
            )
            candidates = utilized_candidates
        else:
            candidates = syntactical_candidates_set

        if add_merged_index:
            merged_indexes = self._add_merged_indexes(candidates, cost_evaluation=cost_evaluation)
            self.logger.debug(f'add merged indexes: {len(merged_indexes)}.')
            self.logger.debug(
                f'merged indexes details:\n' + '\n'.join([str(_) for _ in merged_indexes])
            )

        if filtered:
            candidates = self._filter_candidates(candidates, cost_evaluation)

        self.logger.debug(f'number of candidate indexes: {len(candidates)}.')
        return candidates

    def _generate_candidate_sizes(self, tmp_candidates, filename=None):
        if filename is None:
            filename = f'{self.benchmark}_index_size.json'

        candidates = []

        if os.path.exists(filename):
            self.logger.info(f"load index candidate sizes from {filename}.")

            with open(filename) as f:
                candidate_sizes = json.load(f)

            add_flag = False
            for candidate in tmp_candidates:
                if str(candidate) in candidate_sizes:
                    candidate.size = candidate_sizes[str(candidate)]
                    candidates.append(candidate)

                else:
                    self.logger.debug(f'{candidate}: not in file.')
                    try:
                        self.db_conn.create_index(candidate)
                        self.db_conn.calculate_index_size(candidate)
                        self.db_conn.drop_index(candidate)
                        self.logger.info(f'new candidate {candidate}: \t{candidate.size}MB.')

                        candidate_sizes[str(candidate)] = candidate.size
                        candidates.append(candidate)
                        add_flag = True

                    except Exception as e:
                        self.logger.debug(f'{candidate}: {str(e).strip()}.')

            if add_flag:
                with open(filename, 'w') as f:
                    json.dump(candidate_sizes, f, indent=4)

                self.logger.info(f"save new index candidate sizes to {filename}.")

        else:
            self.logger.info(f"measure all index candidate sizes.")

            candidate_sizes = {}
            for candidate in tmp_candidates:

                try:
                    self.db_conn.create_index(candidate)
                    self.db_conn.calculate_index_size(candidate)
                    self.db_conn.drop_index(candidate)
                    self.logger.debug(f'{candidate}: \t{candidate.size}MB.')

                    candidate_sizes[str(candidate)] = candidate.size
                    candidates.append(candidate)

                    assert candidate.size is not None

                except Exception as e:
                    self.logger.info(f'remove {candidate}: {e}.')

                # self.cost_evaluation.get_index_size(candidate)

            with open(filename, 'w') as f:
                json.dump(candidate_sizes, f, indent=4)
            self.logger.info(f"save index candidate sizes to {filename}.")

        return candidates

    def _add_merged_indexes(self, indexes, cost_evaluation=None):
        if cost_evaluation is None:
            cost_evaluation = self.cost_evaluation

        merged_indexes = []
        index_table_dict = indexes_by_table(indexes)
        for table in index_table_dict:
            for index1, index2 in itertools.permutations(index_table_dict[table], 2):
                merged_index = index_merge(index1, index2)
                if len(merged_index.columns) > self.max_index_width:
                    new_columns = merged_index.columns[: self.max_index_width]
                    merged_index = Index(new_columns)
                if merged_index not in indexes:
                    cost_evaluation.get_index_size(merged_index)
                    indexes.add(merged_index)
                    merged_indexes.append(merged_index)
        return merged_indexes

    def _filter_candidates(self, candidates, cost_evaluation=None):
        if cost_evaluation is None:
            cost_evaluation = self.cost_evaluation

        filtered_candidates = set()
        filtered_num = 0
        for candidate in candidates:
            if cost_evaluation.get_index_size(candidate) > self.index_storage_budget:
                filtered_num += 1
                continue
            filtered_candidates.add(candidate)
        self.logger.info(f'filter {filtered_num} indexes due to storage size.')

        filtered_candidates = set(
            sorted(filtered_candidates, key=lambda x: x)
        )
        return filtered_candidates

    '''
    utils
    '''

    def generate_summary(self):
        evaluation_num = self.cost_evaluation.evaluation_num
        evaluation_duration = self.cost_evaluation.evaluation_duration
        index_modification_duration = self.cost_evaluation.index_modification_duration

        summary_str = (
            f" | BEST PERF: \t{self.best_cost:.2f}"
            f" (imp. {(1 - self.best_cost / self.initial_cost) * 100:.2f}%)\n"
            f" | EVALUATION NUM: \t{evaluation_num}\n"
            f" | TOTAL EVALUATION DURATION: \t{evaluation_duration:.2f} sec."
            f" ({evaluation_duration / self.total_elapsed_time * 100:.2f}%)\n"
            f" | TOTAL INDEX MODIFICATION DURATION: \t{index_modification_duration:.2f} sec."
            f" ({index_modification_duration / self.total_elapsed_time * 100:.2f}%)\n"
            f" | TOTAL ELAPSED TIME: \t{self.total_elapsed_time:.2f} sec."
        )

        self.logger.info(
            f"algorithm summary:\n"
            f"{summary_str}"
        )

        with open(f'summary_{self.task_id}.txt', 'a') as f:
            f.write(f"[{self.method}]\n")
            f.write(summary_str)
            f.write('\n\n')

        return summary_str

    def setup_logger(self, log_file=None):
        if log_file is None:
            log_file = os.path.join(self.logging_dir, f'{self.method}.log')

        setup_logger(output_file=log_file)
        self.logger.info(f'save logs to {log_file}')

    def generate_history(self, history_dir=None):
        if history_dir is None:
            history_dir = self.history_dir
        if not os.path.exists(history_dir):
            os.makedirs(history_dir)
        history_file = os.path.join(history_dir, f"{self.task_id}_{self.method}.json")
        return history_file

    def setup_history_container(self, task_id=None):
        self.logger.info(f'initialize history container: {self.history_file}')
        if self.method in ['BO', 'MFES']:
            container = BOHistoryContainer
        else:
            container = HistoryContainer

        history_container = container(
            task_id=task_id if task_id else self.task_id,
            method=self.method,
            timeout=self.timeout,
            fidelity=self.fidelity,
            index_storage_budget=self.index_storage_budget
        )
        return history_container

    def load_history_container(self, task_id=None, history_file=None, config_space=None, use_bo=False):
        if history_file is None:
            history_file = self.history_file

        if use_bo:
            container = BOHistoryContainer
        else:
            container = HistoryContainer

        history_container = container(
            task_id=task_id if task_id else self.task_id,
            method=self.method,
            timeout=self.timeout,
            fidelity=self.fidelity,
            index_storage_budget=self.index_storage_budget
        )

        history_container.load_history_from_json(history_file, columns=self.columns, config_space=config_space)
        return history_container


class NoIndex(IndexSelectionAlgorithm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_best_index_configuration(self, workload):
        self.workload = workload
        self.initial_cost = self._prepare_index_and_evaluate_cost(workload, [], verbose=True)
        return [], self.initial_cost
