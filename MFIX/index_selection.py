import json
import os
import pdb
import sys
import abc
import pickle
import numpy as np
from abc import abstractmethod
from MFIX.workload import Workload
from MFIX.dbconnector import MysqlConnector, PostgresConnector
from MFIX.benchmark_generation import QueryGenerator, TableGenerator
from MFIX.cost_evaluation import LowFidelityEvaluation, HighFidelityEvaluation
from MFIX.utils.logging_utils import get_logger, setup_logger


DBMS = {'postgres': PostgresConnector, 'mysql': MysqlConnector}


class IndexSelection:
    def __init__(self, args_algo, dbtype, host, port, user, passwd, sock, cnf, dbname,
                 benchmark, scale_factor, queries, pickle_workload,
                 index_storage_budget, eval_timeout,
                 task_id=None, log_dir='log/', history_dir='history/',
                 **kwargs):

        self.task_id = task_id if task_id else f"{benchmark}_SF{scale_factor}_{index_storage_budget}GB"
        self.logging_dir = os.path.join(log_dir, self.task_id)
        self.logger = self.generate_logger()
        self.history_dir = history_dir

        # database-related
        self.dbtype = dbtype
        self.dbname = dbname
        self.db_args = {
            'task_id': task_id,
            'dbtype': dbtype,
            'host': host,
            'port': port,
            'user': user,
            'passwd': passwd,
            'sock': sock,
            'cnf': cnf,
        }
        self.db_connector = None
        self.tables = None
        self.columns = None

        # algorithm-related
        self.args_algo = args_algo
        self.index_storage_budget_GB = index_storage_budget
        self.index_storage_budget = float(index_storage_budget) * 1024
        self.eval_timeout = int(eval_timeout)

        # benchmark-related
        self.benchmark = benchmark
        self.scale_factor = scale_factor
        self.query_ids = eval(queries) if len(queries) else None
        self.workload = None
        self.generate_benchmark(eval(pickle_workload))

    def run_algorithms(self):
        self.setup_db_connector(self.dbname, self.dbtype)

        for algo in self.args_algo:
            try:
                algorithm = self.create_algorithm_object(algo, extra_parameters=self.args_algo[algo])
                algorithm.calculate_best_index_configuration(self.workload)  # set best_configuration, best_cost
                algorithm.generate_summary()
            except Exception as e:
                print(e)

    def create_algorithm_object(self, algorithm_name, extra_parameters):
        if algorithm_name == 'NoIndex':
            from MFIX.algorithms import NoIndex as ALGORITHM

        elif algorithm_name == 'MFES':
            from MFIX.algorithms import MFES as ALGORITHM

        elif algorithm_name == 'BO':
            from MFIX.algorithms import BO as ALGORITHM

        else:
            raise NotImplementedError

        return ALGORITHM(
            task_id=self.task_id,
            logging_dir=self.logging_dir,
            benchmark=self.benchmark,
            db_conn=self.db_connector,
            index_storage_budget=self.index_storage_budget,
            timeout=self.eval_timeout,
            history_dir=self.history_dir,
            tables=self.tables,
            columns=self.columns,
            workload_pickle_file=f"benchmark_pickle/workload_{self.benchmark}_{len(self.query_ids)}_queries.pickle",
            **extra_parameters
        )

    def generate_benchmark(self, pickle_workload=True):
        self.logger.info(f"initialize benchmark: {self.benchmark.upper()}")

        # generate database if needed
        generation_connector = DBMS[self.dbtype](dbname=None, **self.db_args)
        table_generator = TableGenerator(
            self.benchmark, self.scale_factor, generation_connector, self.dbname
        )
        self.tables = table_generator.tables
        self.columns = table_generator.columns

        self.db_connector = DBMS[self.dbtype](dbname=table_generator.database_name, **self.db_args)
        self.db_connector.drop_indexes()
        self.logger.info(f"initial relation size: {round(self.db_connector.get_relation_size(), 2)}MB.")
        self.logger.info(f"initial index size: {round(self.db_connector.get_index_size(), 2)}MB.")

        # self.index_storage_budget = self.index_storage_budget_ratio / 100 * self.db_connector.get_relation_size()
        self.logger.info(f"index storage budget size: {round(self.index_storage_budget/1024, 2)}GB.")

        # generate raw_queries if needed
        pickle_filename = (
            f"benchmark_pickle/workload_{self.benchmark}"
            f"_{len(self.query_ids)}_queries.pickle"
        )
        if pickle_workload and os.path.exists(pickle_filename):
            self.workload = pickle.load(open(pickle_filename, "rb"))
            self.logger.info(f"Load workload from {pickle_filename}")
        else:
            query_generator = QueryGenerator(
                self.benchmark, self.scale_factor, self.db_connector, self.query_ids, self.columns
            )

            if all(
                    os.path.exists(f'queries/{self.benchmark}/{self.query_ids[i]}.sql')
                    for i in range(len(self.query_ids))
            ):
                self.logger.info("Load queries from sql files")
                for query_id in self.query_ids:
                    with open(f'queries/{self.benchmark}/{query_id}.sql') as f:
                        text = f.read()
                    try:
                        query_generator.add_new_query(query_id, text)
                        if self.benchmark == 'job':
                            self.logger.debug(f'add {query_id}')
                        else:
                            self.logger.debug(f'add Q{query_id}')
                    except Exception as e:
                        self.logger.warning(e)
                self.workload = Workload(self.benchmark, query_generator.queries)

            else:
                query_generator.generate()
                self.workload = Workload(self.benchmark, query_generator.queries)
                self.logger.info(f'Generate {len(self.workload.queries)} queries.')

                if not os.path.exists('queries'):
                    os.mkdir('queries')
                if not os.path.exists(f'queries/{self.benchmark}'):
                    os.mkdir(f'queries/{self.benchmark}')
                for query in self.workload.queries:
                    with open(f'queries/{self.benchmark}/{query.query_id}.sql', 'w') as f:
                        f.write(query.query_text)

            if pickle_workload:
                if not os.path.exists('benchmark_pickle'):
                    os.mkdir('benchmark_pickle')
                pickle.dump(self.workload, open(pickle_filename, "wb"))
                self.logger.info(f"Save workload to {pickle_filename}")

    def generate_logger(self):
        if not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)
        log_file = os.path.join(self.logging_dir, f'%s.log' % str(self.task_id))

        setup_logger(output_file=log_file)
        logger = get_logger(self.__class__.__name__)
        return logger

    def setup_db_connector(self, database_name, database_system):
        if self.db_connector:
            self.logger.info("Create new database connector (closing old)")
            self.db_connector.close_connection()
        self.db_connector = DBMS[database_system](dbname=database_name, **self.db_args)
