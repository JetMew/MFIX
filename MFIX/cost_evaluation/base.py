from MFIX.utils.logging_utils import get_logger


class CostEvaluation:
    def __init__(self, db_connector):
        self.db_connector = db_connector
        self.current_indexes = set()
        self.completed = False

        self.index_modification_num = 0
        self.index_modification_duration = 0

        self.evaluation_num = 0
        self.evaluation_duration = 0

        self.logger = get_logger(self.__class__.__name__)

    def complete_cost_evaluation(self):
        raise NotImplementedError

    def reset_cost_evaluation(self):
        self.complete_cost_evaluation()
        self.completed = False

    def calculate_cost(self, workload, indexes, store_size=False, verbose=False):
        return self._calculate_cost(workload, indexes, store_size, verbose)

    def calculate_reward(self, workload, indexes, store_size=False, table_scan_time_length=1000, init=False):
        return self._calculate_reward(workload, indexes, store_size, table_scan_time_length, init)

    def get_current_size_total(self):
        return self._get_current_size_total()

    def get_index_size(self, index):
        return self._get_index_size(index)

    def _calculate_cost(self, workload, indexes, store_size=False, verbose=False):
        raise NotImplementedError

    def _calculate_reward(self, workload, indexes, store_size=False, table_scan_time_length=1000, init=False):
        raise NotImplementedError

    def _get_current_size_total(self):
        raise NotImplementedError

    def _get_index_size(self, index):
        raise NotImplementedError

    def _prepare_cost_calculation(self, indexes, store_size=False):
        raise NotImplementedError

    def _get_cost(self, query):
        raise NotImplementedError

    def _get_sub_tree_cost(self, query, timeout=None):
        raise NotImplementedError

    def summary(self):
        self.logger.info(
            f"cost evaluation summary:"
            f"\n\tevaluation numbers: {self.evaluation_num} sec."
            f"\n\ttotal evaluation duration: {round(self.evaluation_duration, 2)} sec."
            f"\n\ttotal index modification duration: {round(self.index_modification_duration, 2)} sec."
        )
