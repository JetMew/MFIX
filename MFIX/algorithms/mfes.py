import os
import pdb
import time
import numpy as np
from tqdm import tqdm
from MFIX.algorithms.bo import BO
from MFIX.cost_evaluation import LowFidelityEvaluation, HighFidelityEvaluation


class MFES(BO):
    def __init__(
            self,
            max_runtime_minutes_low=60,
            max_runtime_minutes_high=600,
            load_history=False,
            load_history_files=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.max_time_low = int(max_runtime_minutes_low) * 60
        self.max_time_high = int(max_runtime_minutes_high) * 60

        self.load_history = bool(load_history)
        self.load_history_files = eval(load_history_files) if len(load_history_files) else None
        if self.load_history_files is None:
            self.load_history_files = [self.history_file.replace('MFES', 'BO_fidelity0'), ]
        for load_history_file in load_history_files:
            assert os.path.exists(os.path.join(self.history_dir, load_history_file)), f"{load_history_file} not exists"

    def _calculate_best_index_configuration(self, workload):
        self.workload = workload

        source_data = self.run_low_fidelity()
        return self.run_high_fidelity(source_data)

    def run_low_fidelity(self):
        self.logger.info('Stage 1: pre-train on low fidelity.')
        self.fidelity = 0

        self.logger.info(f"evaluate with WHAT-IF CALLS.")
        self.cost_evaluation = LowFidelityEvaluation(self.db_conn)
        self.setup_config_space()

        source_data = []

        if self.load_history:
            for i, file in enumerate(self.load_history_files):
                history_container = self.load_history_container(
                    task_id=f'{self.task_id}_0',
                    history_file=self.load_history_files[i],
                    config_space=self.config_space,
                    use_bo=True
                )
                source_data.append(history_container.data)

        else:
            self.setup_bo_basics()
            self.initial_configurations = self.create_initial_design(self.init_strategy, init_num=self.init_num)
            self.history_container = self.setup_history_container(
                task_id=f'{self.task_id}'.replace('MFES', 'BO_fidelity0'))
            self.history_file = self.generate_history()

            global_start_time_low = time.time()
            for _ in tqdm(range(self.max_runs)):
                self.iterate()
                consumed_time = time.time() - global_start_time_low
                if consumed_time > self.max_time_low:
                    self.logger.info(
                        f"Stopping after {_ + 1} iterations because of timing constraints."
                    )
                    break
            source_data.append(self.history_container.data)

        return source_data

    def run_high_fidelity(self, source_data):
        self.logger.info('Stage 2: fine-tune on high fidelity.')
        self.iteration_id = 0
        self.fidelity = 1
        self.surrogate_type = 'rgpe_prf'

        configs = [_ for data in source_data for _ in list(data.keys())]
        perfs = [_ for data in source_data for _ in list(data.values())]
        best_config = configs[np.argmin(perfs)]

        self.cost_evaluation = HighFidelityEvaluation(self.db_conn, timeout=int(self.timeout))
        self.logger.info(f"evaluate with QUERY EXECUTION.")

        self.setup_bo_basics(source_hpo_data=source_data)
        self.initial_configurations = self.create_initial_design(
            init_strategy=self.init_strategy,
            init_num=self.init_num,
            default_config=best_config
        )
        self.history_container = self.setup_history_container(task_id=f'{self.task_id}')
        self.history_file = self.generate_history()

        if os.path.exists(self.history_file):
            self.history_container.load_history_from_json(self.history_file, self.config_space, self.columns)

        global_start_time_high = time.time()

        for _ in tqdm(range(self.max_runs)):
            self.iterate()
            consumed_time = time.time() - global_start_time_high
            if consumed_time > self.max_time_high:
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
