import pdb
import re
import time
import sqlparse
import psycopg2
import xml.etree.ElementTree as ET
from .base import DBConnector


class PostgresConnector(DBConnector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dbtype = "postgres"
        if not self.dbname:
            self.dbname = 'postgres'

        self.create_connection()
        self.set_random_seed()
        self.logger.debug("initialize postgres connector")

    '''
    basic
    '''

    def create_connection(self):
        if self._conn:
            self._conn.close()
            self.logger.debug("Database connector closed: {}".format(self.dbname))

        self._conn = psycopg2.connect(host=self.host,
                                      user=self.user,
                                      password=self.passwd,
                                      database=self.dbname,
                                      port=self.port)
        self._conn.autocommit = self.autocommit
        self._cursor = self._conn.cursor()
        self.logger.debug(f"Database connector created: {self.dbname}")

    def get_databases(self):
        result = self.execute_fetch("select datname from pg_database", False)
        return [x[0] for x in result]

    def create_database(self, database_name):
        self.execute("create database {}".format(database_name))
        self.logger.info("Database {} created".format(database_name))

    def drop_database(self, database_name):
        statement = f"DROP DATABASE {database_name};"
        self.execute(statement)
        self.logger.info(f"Database {database_name} dropped")

    def database_exists(self, database_name):
        statement = (
            "SELECT EXISTS ( "
            "SELECT 1 "
            "FROM pg_database "
            f"WHERE datname = '{database_name}');"
        )
        result = self.execute_fetch(statement)
        return result[0]

    def table_exists(self, table_name):
        statement = (
            "SELECT EXISTS ( "
            "SELECT 1 "
            "FROM pg_tables "
            f"WHERE tablename = '{table_name}');"
        )
        result = self.execute_fetch(statement)
        return result[0]

    def import_data(self, table, path, delimiter="|"):
        with open(path, "r") as file:
            self._cursor.copy_from(file, table, sep=delimiter, null="")

    def analyze(self):
        self.logger.debug(f"Postgres: `analyze`")
        self.execute("analyze")

    def set_random_seed(self, value=0.17):
        self.logger.debug(f"Postgres: Set random seed `SELECT setseed({value})`")
        self.execute(f"SELECT setseed({value})")

    def get_table_row_count(self, table_name):
        statement = (
            f"SELECT count(*) FROM {table_name};"
        )
        result = self.execute_fetch(statement)
        return result[0]

    def get_all_columns(self):
        statement = (
            "SELECT table_name, column_name "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "ORDER BY table_name, column_name;"
        )
        result = self.execute_fetch(statement)

        columns_dict = {}
        for table_name, column_name in result:
            if table_name not in columns_dict:
                columns_dict[table_name] = []
            columns_dict[table_name].append(column_name)
        return columns_dict

    def get_column_selectivity(self, column):
        statement = (
            f"SELECT COUNT(DISTINCT({column.name})) "
            f"FROM {column.table.name};"
        )
        result = self.execute_fetch(statement)[0]
        return float(result) / column.table.row_count

    def get_relation_size(self):  # size in MB
        statement = (
            "select sum(pg_relation_size(table_name::text))/1024/1024 from "
            "(select table_name from information_schema.tables "
            "where table_schema='public') as all_tables"
        )
        result = self.execute_fetch(statement)[0]
        return float(result)

    def get_index_size(self):  # size in MB
        statement = (
            "select sum(pg_indexes_size(table_name::text))/1024/1024 from "
            "(select table_name from information_schema.tables "
            "where table_schema='public') as all_tables"
        )
        result = self.execute_fetch(statement)[0]
        return float(result)

    '''
    index-related
    '''

    def get_indexes(self):
        statement = (
            "SELECT tablename, indexname "
            "FROM pg_indexes "
            "WHERE schemaname = 'public' "
            f"AND table_catalog = '{self.dbname}' "
            "ORDER BY tablename, indexname;"
        )
        result = self.execute_fetch(statement)
        return result[0]

    def create_index(self, index):
        table_name = index.table()
        statement = (
            f"create index {index.index_idx()} "
            f"on {table_name} ({index.joined_column_names()})"
        )
        self.execute(statement)
        if not index.size:
            size = self.execute_fetch(
                f"select relpages from pg_class c "
                f"where c.relname = '{index.index_idx()}'"
            )
            size = size[0]
            index.size = size * 8 * 1024

    def drop_index(self, index):
        statement = f"drop index {index.index_idx()}"
        self.execute(statement)

    def drop_indexes(self):
        statement = "select indexname from pg_indexes where schemaname='public'"
        indexes = self.execute_fetch(statement, one=False)
        constraint_names = self.get_constraints()
        for index in indexes:
            index_name = index[0]
            if index_name in constraint_names:
                continue
            drop_stmt = "drop index {}".format(index_name)
            self.logger.debug("Dropping index {}".format(index_name))
            self.execute(drop_stmt)

    def get_constraints(self):
        statement = (
            "SELECT kcu.constraint_name "
            "FROM information_schema.table_constraints tco "
            "JOIN information_schema.key_column_usage kcu "
            "ON kcu.constraint_name = tco.constraint_name "
            "AND kcu.constraint_schema = tco.constraint_schema "
            "AND kcu.constraint_name = tco.constraint_name "
            "WHERE tco.constraint_type = 'PRIMARY KEY' "
            "OR tco.constraint_type = 'FOREIGN KEY';"
        )
        indexes = self.execute_fetch(statement, one=False)
        return [_[0] for _ in indexes]

    def calculate_index_size(self, index):  # size in MB
        statement = (
            f"select relpages from pg_class c "
            f"where c.relname = '{index.index_idx()}'"
        )
        size = self.execute_fetch(statement)[0]
        index.size = size * 8 / 1024  # 8kB / page
        return size

    '''
    hypo-index-related
    '''

    def enable_simulation(self):
        self.execute("DROP EXTENSION IF EXISTS hypopg;")
        self.execute("CREATE EXTENSION hypopg;")
        self.commit()
        self.logger.debug('enable hypopg.')

    def disable_simulation(self):
        self.execute("DROP EXTENSION IF EXISTS hypopg;")
        self.commit()

    def get_simulated_indexes(self):
        statement = "select * from hypopg_list_indexes;"
        indexes = self.execute_fetch(statement, one=False)
        return indexes

    def simulate_index(self, index):
        table_name = index.table()
        statement = (
            "select * from hypopg_create_index( "
            f"'create index on {table_name} "
            f"({index.joined_column_names()})')"
        )
        result = self.execute_fetch(statement)
        return result

    def drop_simulated_index(self, oid):
        statement = f"select * from hypopg_drop_index({oid})"
        result = self.execute_fetch(statement)
        assert result[0] is True, f"Could not drop simulated index with oid = {oid}."

    def drop_simulated_indexes(self):
        statement = "SELECT * FROM hypopg_reset();"
        self.execute(statement)
        self.logger.debug('Hypopg reset: drop all hypothetical indexes!')

    def estimate_index_size(self, potential_index):  # size in MB
        index_oid = potential_index.hypopg_oid
        statement = f"select hypopg_relation_size({index_oid});"
        result = self.execute_fetch(statement)[0] / 1024 / 1024
        assert result > 0, "Hypothetical index does not exist."
        potential_index.estimated_size = result
        return result

    '''
    query-related
    '''

    def execute_query(self, query, timeout=None):
        conn = psycopg2.connect(
            host=self.host,
            user=self.user,
            password=self.passwd,
            database=self.dbname,
            port=self.port,
            options=f'-c statement_timeout={timeout}s'
        )
        conn.autocommit = True
        cur = conn.cursor()

        try:
            start_time = time.time()
            cur.execute(query.query_text)
            cur.fetchall()
            elapsed_time = time.time() - start_time

        except Exception as e:
            error_str = str(e).strip('\n')
            self.logger.warning(f"{query.query_id}: {error_str}")
            elapsed_time = timeout

        finally:
            cur.close()
            conn.close()

        return elapsed_time

    def explain_analyze_query(self, query, timeout=None):
        query_text = self._prepare_query(query)

        # NOTE: can be multiple selects
        query_plans = []
        query_text = sqlparse.format(query_text, strip_comments=True)
        stmts = sqlparse.split(query_text)
        for stmt in stmts:
            conn = psycopg2.connect(
                host=self.host,
                user=self.user,
                password=self.passwd,
                database=self.dbname,
                port=self.port,
                options=f'-c statement_timeout={timeout}s'
            )
            conn.autocommit = True
            cur = conn.cursor()

            try:
                statement = f"explain (format xml, analyze) {stmt}"
                cur.execute(statement)
                query_plan = cur.fetchone()[0]
                query_plans.append(query_plan)

            except Exception as e:
                self.logger.warning(f"{query.query_id}: {e}")

            finally:
                cur.close()
                conn.close()

        self._cleanup_query(query)
        return query_plans

    def _get_sub_tree_cost(self, query, fidelity=0, timeout=None):
        total_cost = 0
        index_usage = []
        non_index_usage = []

        if fidelity == 0:
            xml_strings = self._get_plan(query, plan_format='xml')
            cost_name = 'Total-Cost'
            cost_scale = 1
        elif fidelity == 1:
            xml_strings = self.explain_analyze_query(query, timeout)
            cost_name = 'Actual-Total-Time'
            cost_scale = 1000

            if len(xml_strings) == 0:
                # timeout query
                return timeout, [], []
        else:
            raise ValueError("fidelity must be 0 or 1.")

        for xml_string in xml_strings:
            root = ET.fromstring(xml_string)

            ns = {'explain': 'http://www.postgresql.org/2009/explain'}
            total_cost += (float(root[0][0].find(f'explain:{cost_name}', ns).text) / cost_scale)

            sub_plans = root.findall('.//explain:Plan', ns)
            for sub_plan in sub_plans:
                # node_type = sub_plan.find('explain:Node-Type', ns)
                table_name = sub_plan.find('explain:Relation-Name', ns)
                index_name = sub_plan.find('explain:Index-Name', ns)
                cost = float(sub_plan.find(f'explain:{cost_name}', ns).text) / cost_scale

                if table_name is None:
                    continue

                if index_name is None:
                    non_index_usage.append({
                        'table_name': table_name.text,
                        'cost': cost
                    })
                    # non_index_usage.append([table_name.text, cost])
                else:
                    index_usage.append({
                        'index_name': index_name.text,
                        'table_name': table_name.text,
                        'cost': cost
                    })
                    # index_usage.append([index_name.text, cost])

        return total_cost, index_usage, non_index_usage

    def _get_cost(self, query):
        total_cost = 0
        query_plans = self._get_plan(query)
        for query_plan in query_plans:
            total_cost += query_plan["Total Cost"]
        return total_cost

    def _get_plan(self, query, plan_format='json'):
        plan_format = plan_format.lower()
        query_text = self._prepare_query(query)

        # NOTE: can be multiple selects
        query_plans = []
        query_text = sqlparse.format(query_text, strip_comments=True)
        stmts = sqlparse.split(query_text)
        for stmt in stmts:
            statement = f"explain (format {plan_format}) {stmt}"
            if plan_format == 'json':
                query_plan = self.execute_fetch(statement)[0][0]["Plan"]
            else:
                query_plan = self.execute_fetch(statement)[0]
            query_plans.append(query_plan)

        self._cleanup_query(query)
        return query_plans

    def update_query_text(self, text):
        text = text.replace(";\nlimit ", " limit ").replace("limit -1", "")
        text = re.sub(r" ([0-9]+) days\)", r" interval '\1 days')", text)
        text = re.sub(r" day \(\d+\)", r" day", text)
        text = self._add_alias_subquery(text)
        return text

    def _cleanup_query(self, query):
        for query_statement in query.query_text.split(";"):
            if "drop view" in query_statement:
                self.execute(query_statement)
                self.commit()

    def _add_alias_subquery(self, query_text):
        # PostgresSQL requires an alias for subqueries
        text = query_text.lower()
        positions = []
        for match in re.finditer(r"((from)|,)[  \n]*\(", text):
            counter = 1
            pos = match.span()[1]
            while counter > 0:
                char = text[pos]
                if char == "(":
                    counter += 1
                elif char == ")":
                    counter -= 1
                pos += 1
            next_word = query_text[pos:].lstrip().split(" ")[0].split("\n")[0]
            if next_word[0] in [")", ","] or next_word in [
                "limit",
                "group",
                "order",
                "where",
            ]:
                positions.append(pos)
        for pos in sorted(positions, reverse=True):
            query_text = query_text[:pos] + " as alias123 " + query_text[pos:]
        return query_text
