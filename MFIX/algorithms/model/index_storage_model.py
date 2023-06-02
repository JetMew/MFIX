import numpy as np
from MFIX.utils.config_space.util import convert_configurations_to_array


class IndexStorageModel:
    def __init__(
            self,
            candidates,
            index_storage_budget: float,
            config_space
    ):
        self.index_storage_budget = index_storage_budget
        self.candidates = candidates

        hps = config_space.get_hyperparameters()
        self.candidate_sizes = np.zeros(len(hps))
        for i, hp in enumerate(hps):
            candidate = next((c for c in self.candidates if hp.name == str(c)))
            self.candidate_sizes[i] = candidate.size

    def p_feasible(self, X: np.ndarray):
        cost = np.dot(X, self.candidate_sizes)
        p = np.array((cost < self.index_storage_budget), dtype=np.float64)
        p = np.reshape(p, (-1, 1))
        return p

    def get_cost(self, config):
        X = convert_configurations_to_array([config, ])
        cost = np.dot(X, self.candidate_sizes)
        return np.asscalar(cost)
