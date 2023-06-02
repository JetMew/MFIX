import json
import time
import threading
import subprocess
from .base import CostEvaluation
from queue import Queue


class HighFidelityEvaluation(CostEvaluation):
    def __init__(self, db_connector, timeout=None):
        super().__init__(db_connector)

        self.timeout = timeout if timeout else 600
        self.logger.info(f"evaluation timeout: {self.timeout}")
        self.results = Queue()

    def complete_cost_evaluation(self):
        self.completed = True

        for index in self.current_indexes.copy():
            self._drop_index(index)

        assert self.current_indexes == set()

    def _calculate_cost(self, workload, indexes, store_size=False, verbose=False):
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."

        index_modify_time = self._prepare_cost_calculation(indexes, store_size=store_size)

        assert (
            self.results.empty()
        ), "Unhandled results."

        start_time = time.time()

        threads = []
        for query in workload.queries:
            thread = threading.Thread(
                target=self._get_cost,
                args=(query,)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        end_time = time.time()
        evaluation_time = end_time - start_time

        self.evaluation_num += 1
        self.evaluation_duration += evaluation_time

        cost_per_query = {}
        while not self.results.empty():
            query, cost = self.results.get()
            cost_per_query[query.query_id] = cost

        total_cost = sum([_ for _ in cost_per_query.values()])
        latency = max([_ for _ in cost_per_query.values()])

        if verbose:
            self.logger.info(
                f"\n  total cost: {round(total_cost, 2)} sec. "
                f"\n  elapsed time: {round(latency, 2)} sec."
                f"\n  number of queries: {len(cost_per_query)}"
            )
        self.logger.debug('elapsed times per query:' + json.dumps(cost_per_query, indent=2))

        return total_cost, cost_per_query, index_modify_time

    def _calculate_reward(self, workload, indexes, store_size=False, table_scan_time_length=1000, init=False):
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        index_modify_time = self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0
        rewards = {}

        if init:
            timeout = 60000
        else:
            timeout = self.timeout

        assert (
            self.results.empty()
        ), "Unhandled results."

        start_time = time.time()

        threads = []
        for query in workload.queries:
            thread = threading.Thread(
                target=self._get_sub_tree_cost,
                args=(query, timeout)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        end_time = time.time()
        evaluation_time = end_time - start_time

        self.evaluation_num += 1
        self.evaluation_duration += evaluation_time

        elapsed_times = {}
        while not self.results.empty():
            query, cost, index_usage, non_index_usage = self.results.get()

            total_cost += cost
            elapsed_times[query.query_id] = cost

            if non_index_usage:
                for table_scan in non_index_usage:
                    table_name = table_scan['table_name']
                    cost = table_scan['cost']

                    if init:
                        query.table_scan_times[table_name] = max(cost, query.table_scan_times.get(table_name, 0))

            if index_usage:
                for index_scan in index_usage:
                    index_name = index_scan['index_name']
                    table_name = index_scan['table_name']
                    cost = index_scan['cost']

                    if init:
                        query.table_scan_times[table_name] = max(cost, query.table_scan_times.get(table_name, 0))

                    index = None
                    for _index in indexes:
                        if _index.index_idx() == index_name:
                            index = _index
                    if index:
                        if index not in rewards:
                            rewards[index] = 0
                        rewards[index] += (query.table_scan_times[table_name] - cost)

        if len(elapsed_times) < len(workload.queries):
            check_cmd = """ps -ef|grep EXPLAIN|grep -v grep|cut -c 10-16"""
            clear_cmd = """ps -ef|grep EXPLAIN|grep -v grep|cut -c 10-16|xargs kill -9"""
            result = subprocess.run(check_cmd, check=True, shell=True, capture_output=True, text=True, close_fds=True).stdout
            if init:
                assert result == '', 'initial run must finish.'
            if result != '':
                subprocess.check_call(clear_cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, close_fds=True)
                self.logger.info('clear unfinished postgres EXPLAIN.')

        self.logger.debug('elapsed times per query:' + json.dumps(elapsed_times, indent=2))
        return total_cost, index_modify_time, rewards

    def _get_current_size_total(self):
        total = 0.
        for index in self.current_indexes:
            if not index.size:
                self.db_connector.calculate_index_size(index)
            total += index.size

        return total

    def _get_index_size(self, index):
        if index.size:
            return index.size

        _index = None
        for _ in self.current_indexes:
            if index == _:
                _index = _
                break

        if _index:
            if not _index.size:
                self.db_connector.calculate_index_size(index)
        else:
            self._create_index(index, store_size=True)
            self._drop_index(index)

        return index.size

    def _prepare_cost_calculation(self, indexes, store_size=False):
        start_time = time.time()

        invalid_indexes = []

        for index in set(indexes) - self.current_indexes:
            flag = self._create_index(index, store_size=store_size)
            if not flag:
                invalid_indexes.append(index)
        for index in self.current_indexes - set(indexes):
            self._drop_index(index)

        self.db_connector.analyze()

        index_modify_time = time.time() - start_time

        assert self.current_indexes == set(indexes) - set(invalid_indexes)
        return index_modify_time

    def _get_cost(self, query):
        elapsed_time = self.db_connector.execute_query(query, timeout=self.timeout)
        self.results.put((query, elapsed_time))

    def _get_sub_tree_cost(self, query, timeout=None):
        if timeout is None:
            timeout = self.timeout

        cost, index_usage, non_index_usage = self.db_connector.get_sub_tree_cost(query, fidelity=1, timeout=timeout)
        self.results.put((query, cost, index_usage, non_index_usage))

    def _create_index(self, potential_index, store_size=False):
        try:
            start_time = time.time()
            self.db_connector.create_index(potential_index)
            end_time = time.time()

            self.index_modification_num += 1
            self.index_modification_duration += end_time - start_time

            self.current_indexes.add(potential_index)

            if store_size:
                self.db_connector.calculate_index_size(potential_index)
            return True

        except Exception as e:
            self.logger.warning(f"cannot create index {potential_index}: {e}")
            return False

    def _drop_index(self, index):
        try:
            start_time = time.time()
            self.db_connector.drop_index(index)
            end_time = time.time()

            self.index_modification_num += 1
            self.index_modification_duration += end_time - start_time

            self.current_indexes.remove(index)

        except Exception as e:
            self.logger.warning(f"cannot drop index {index}: {e}")