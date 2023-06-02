from MFIX.utils.logging_utils import get_logger


class DBConnector:
    def __init__(self, task_id, dbtype, host, port, user, passwd, sock, cnf,
                 dbname=None, autocommit=True,
                 **kwargs):
        self.dbtype = dbtype
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.dbname = dbname
        self.sock = None if len(sock) == 0 else sock
        self.cnf = cnf
        self.autocommit = autocommit

        self.task_id = task_id
        self.logger = get_logger(self.__class__.__name__)

        self._conn = None
        self._cursor = None

    '''
    basic
    '''

    def create_connection(self):
        raise NotImplementedError

    def close_connection(self):
        if self._conn:
            self._conn.close()
            self.logger.debug("Database connector closed: {}".format(self.dbname))

    def execute(self, sql):
        self._cursor.execute(sql)

    def execute_fetch(self, sql, one=True):
        self._cursor.execute(sql)
        if one:
            return self._cursor.fetchone()
        return self._cursor.fetchall()

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def get_databases(self):
        raise NotImplementedError

    def create_database(self, database_name):
        raise NotImplementedError

    def drop_database(self, database_name):
        raise NotImplementedError

    def database_exists(self, database_name):
        raise NotImplementedError

    def table_exists(self, table_name):
        raise NotImplementedError

    def import_data(self, table, path, delimiter='|'):
        raise NotImplementedError

    def analyze(self):
        raise NotImplementedError

    def set_random_seed(self, value=0.17):
        raise NotImplementedError

    def supports_index_simulation(self):
        if self.dbtype == "postgres":
            return True
        return False

    def get_table_row_count(self, table_name):
        raise NotImplementedError

    def get_all_columns(self):
        raise NotImplementedError

    def get_column_selectivity(self, column):
        raise NotImplementedError

    def get_relation_size(self):
        raise NotImplementedError

    def get_index_size(self):
        raise NotImplementedError

    '''
    index-related
    '''
    # def get_indexes(self):
    #     raise NotImplementedError
    #
    # def get_indexes_size(self):
    #     raise NotImplementedError

    def create_index(self, index):
        raise NotImplementedError

    def drop_index(self, index):
        raise NotImplementedError

    def drop_indexes(self):
        raise NotImplementedError

    def calculate_index_size(self, index):
        raise NotImplementedError

    '''
    query-related
    '''

    def execute_query(self, query, timeout=None):
        raise NotImplementedError

    def get_cost(self, query):
        cost = self._get_cost(query)
        return cost

    def get_sub_tree_cost(self, query, fidelity=0, timeout=None):
        total_cost, index_usage, non_index_usage = self._get_sub_tree_cost(query, fidelity=fidelity, timeout=timeout)
        return total_cost, index_usage, non_index_usage

    def _get_cost(self, query):
        raise NotImplementedError

    def _get_sub_tree_cost(self, query, fidelity=0, timeout=None):
        raise NotImplementedError

    def get_plan(self, query, plan_format='json'):
        plans = self._get_plan(query, plan_format=plan_format)
        return plans

    def _get_plan(self, query, plan_format='json'):
        raise NotImplementedError

    def _prepare_query(self, query):
        for query_statement in query.query_text.split(";"):
            if "create view" in query_statement:
                try:
                    self.execute(query_statement)
                except Exception as e:
                    self.logger.error(e)
            elif "select" in query_statement or "SELECT" in query_statement:
                return query_statement

    def update_query_text(self, text):
        raise NotImplementedError

    def _add_alias_subquery(self, query_text):
        raise NotImplementedError

    def _cleanup_query(self, query):
        raise NotImplementedError
