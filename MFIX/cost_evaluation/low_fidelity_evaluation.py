import pdb
import time
from .base import CostEvaluation


class LowFidelityEvaluation(CostEvaluation):
    def __init__(self, db_connector):
        super().__init__(db_connector)

        self.simulated_indexes = {}
        self.cost_requests = 0
        self.cache_hits = 0
        self.cache = {}     # {(query_object, relevant_indexes): cost}
        self.relevant_indexes_cache = {}

        self.db_connector.enable_simulation()

    def complete_cost_evaluation(self):
        self.completed = True

        for index in self.current_indexes.copy():
            self._drop_simulated_index(index)

        assert self.current_indexes == set()

    def _calculate_cost(self, workload, indexes, store_size=False, verbose=False):
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        index_modify_time = self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0
        cost_per_query = {}

        start_time = time.time()
        for query in workload.queries:
            self.cost_requests += 1
            cost = self._request_cache(query, indexes)
            cost_per_query[query.query_id] = cost
            total_cost += cost
            # total_cost += self._request_cache(query, indexes)  # with cache
            # total_cost += self._get_cost(query)  # without cache
        end_time = time.time()
        evaluation_time = end_time - start_time

        self.evaluation_num += 1
        self.evaluation_duration += evaluation_time

        return total_cost, cost_per_query, index_modify_time

    def _calculate_reward(self, workload, indexes, store_size=False, table_scan_time_length=1000, init=False):
        assert (
                self.completed is False
        ), "Cost Evaluation is completed and cannot be reused."
        index_modify_time = self._prepare_cost_calculation(indexes, store_size=store_size)
        total_cost = 0
        rewards = {}

        start_time = time.time()
        for query in workload.queries:
            self.cost_requests += 1
            cost, index_usage, non_index_usage = self._get_sub_tree_cost(query)
            total_cost += cost
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
                        if _index.hypopg_name == index_name:
                            index = _index
                    if index:
                        if index not in rewards:
                            rewards[index] = 0
                        rewards[index] += (query.table_scan_times[table_name] - cost)

        end_time = time.time()
        evaluation_time = end_time - start_time

        self.evaluation_num += 1
        self.evaluation_duration += evaluation_time

        return total_cost, index_modify_time, rewards

    def _get_current_size_total(self):
        total = 0.
        for index in self.current_indexes:
            if not index.estimated_size:
                self.db_connector.estimate_index_size(index)
            total += index.estimated_size

        return total

    def _get_index_size(self, index):
        if index.estimated_size:
            return index.estimated_size

        _index = None
        for _ in self.current_indexes:
            if index == _:
                _index = _
                break

        if _index:
            if not _index.estimated_size:
                self.db_connector.estimate_index_size(index)
            else:
                index.estimated_size = _index.estimated_size
        else:
            self._simulate_index(index, store_size=True)
            self._drop_simulated_index(index)

        return index.estimated_size

    def _prepare_cost_calculation(self, indexes, store_size=False):
        start_time = time.time()

        for index in set(indexes) - self.current_indexes:
            self._simulate_index(index, store_size=store_size)
        for index in self.current_indexes - set(indexes):
            self._drop_simulated_index(index)

        index_modify_time = time.time() - start_time

        assert self.current_indexes == set(indexes)
        return index_modify_time

    def _get_cost(self, query):
        return self.db_connector.get_cost(query)

    def _get_sub_tree_cost(self, query, timeout=None):
        return self.db_connector.get_sub_tree_cost(query, fidelity=0)

    def _simulate_index(self, potential_index, store_size=False):
        start_time = time.time()
        result = self.db_connector.simulate_index(potential_index)
        end_time = time.time()

        self.index_modification_num += 1
        self.index_modification_duration += end_time - start_time

        index_oid = result[0]
        index_name = result[1]
        self.simulated_indexes[index_oid] = index_name
        potential_index.hypopg_name = index_name
        potential_index.hypopg_oid = index_oid

        if store_size:
            self.db_connector.estimate_index_size(potential_index)

        self.current_indexes.add(potential_index)

    def _drop_simulated_index(self, index):
        start_time = time.time()
        self.db_connector.drop_simulated_index(index.hypopg_oid)
        end_time = time.time()

        self.index_modification_num += 1
        self.index_modification_duration += end_time - start_time

        del self.simulated_indexes[index.hypopg_oid]

        self.current_indexes.remove(index)

    def _request_cache(self, query, indexes):
        q_i_hash = (query, frozenset(indexes))
        if q_i_hash in self.relevant_indexes_cache:
            relevant_indexes = self.relevant_indexes_cache[q_i_hash]
        else:
            relevant_indexes = self._relevant_indexes(query, indexes)
            self.relevant_indexes_cache[q_i_hash] = relevant_indexes

        # Check if query and corresponding relevant indexes in cache
        if (query, relevant_indexes) in self.cache:
            self.cache_hits += 1
            return self.cache[(query, relevant_indexes)]
        # If no cache hit request cost from database system
        else:
            cost = self._get_cost(query)
            self.cache[(query, relevant_indexes)] = cost
            return cost

    @staticmethod
    def _relevant_indexes(query, indexes):
        relevant_indexes = [
            x for x in indexes if any(c in query.indexable_columns for c in x.columns)
        ]
        return frozenset(relevant_indexes)

    # def which_indexes_utilized_and_cost(self, query, indexes):
    #     self._prepare_cost_calculation(indexes, store_size=True)
    #     plans = self.db_connector.get_plan(query)
    #
    #     cost = 0
    #     recommended_indexes = set()
    #
    #     for plan in plans:
    #         cost += plan["Total Cost"]
    #         plan_str = str(plan)
    #
    #         for index in self.current_indexes:
    #             assert (
    #                     index in indexes
    #             ), "Something went wrong with _prepare_cost_calculation."
    #             if index.hypopg_name not in plan_str:
    #                 continue
    #             recommended_indexes.add(index)
    #
    #     return recommended_indexes, cost

    def which_indexes_utilized_and_cost(self, workload, indexes):
        self._prepare_cost_calculation(indexes, store_size=True)

        cost = 0
        recommended_indexes = set()

        for query in workload.queries:
            plans = self.db_connector.get_plan(query)

            for plan in plans:
                cost += plan["Total Cost"]
                plan_str = str(plan)

                for index in self.current_indexes:
                    assert (
                            index in indexes
                    ), "Something went wrong with _prepare_cost_calculation."
                    if index.hypopg_name not in plan_str:
                        continue
                    recommended_indexes.add(index)

        return recommended_indexes, cost

    def _log_cache_hits(self):
        hits = self.cache_hits
        requests = self.cost_requests
        self.logger.debug(f"Total cost cache hits:\t{hits}")
        self.logger.debug(f"Total cost requests:\t\t{requests}")
        if requests == 0:
            return
        ratio = round(hits * 100 / requests, 2)
        self.logger.debug(f"Cost cache hit ratio:\t{ratio}%")
