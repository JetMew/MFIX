import logging
import pdb

import sqlparse
import sql_metadata
from typing import List
from sqlparse.sql import Comparison, Identifier, Function
from MFIX.index import Index
from MFIX.utils.logging_utils import get_logger


class Query:
    def __init__(self, query_id, query_text, columns=None):
        self.query_id = query_id
        self.query_text = query_text

        if columns is None:
            self.indexable_columns = []
        else:
            self.indexable_columns = columns

        self._columns_dict = None
        self._tables_aliases = None
        self._indexable_columns_per_clause = None

    def __repr__(self):
        return f"Q{self.query_id}"

    def indexable_columns_per_clause(self):
        return self._indexable_columns_per_clause

    def get_indexable_columns_per_clause(self):
        if self._indexable_columns_per_clause:
            return self._indexable_columns_per_clause

        self.logger = get_logger(str(self))
        self.logger.debug('Generate indexable columns per clause.')

        assert len(self.indexable_columns) > 0, "Need indexable columns of the query."

        columns_per_clause = {
            'select': set(),
            'group_by': set(),
            'order_by': set(),
            'filter': set(),
            'join': set()
        }

        query_text = sqlparse.format(self.query_text, strip_comments=True)
        if 'create view' in query_text:
            stmts = sqlparse.split(query_text)
            for i, stmt in enumerate(stmts):

                if 'create view' in stmt:
                    stmt = stmt.split('\n', 1)[1]
                if 'drop view' in stmt:
                    continue

                tables_aliases = None
                columns_dict = None
                try:
                    parser = sql_metadata.Parser(stmt)
                    tables_aliases = parser.tables_aliases
                    columns_dict = parser.columns_dict
                except Exception as e:
                    logging.warning(f'Cannot parse {self.query_id}: {stmt}\n{e}')

                if columns_dict:
                    if not self._tables_aliases:
                        self._tables_aliases = tables_aliases
                    else:
                        for k, v in tables_aliases.items():
                            if k not in self._tables_aliases:
                                self._tables_aliases[k] = v

                    if not self._columns_dict:
                        self._columns_dict = columns_dict
                    else:
                        for k, v in columns_dict.items():
                            if k not in self._columns_dict:
                                self._columns_dict[k] = v
                            else:
                                for _ in v:
                                    if _ not in self._columns_dict[k]:
                                        self._columns_dict[k].append(_)

        else:
            try:
                parser = sql_metadata.Parser(query_text)
                self._tables_aliases = parser.tables_aliases
                self._columns_dict = parser.columns_dict
            except Exception as e:
                logging.warning(f'Cannot parse {self.query_id}, {e}')

        if self._columns_dict:
            for k, v in self._columns_dict.items():
                if k in ['group_by', 'order_by', 'select']:
                    for identifier in v:
                        found, column = self.find_column(identifier)
                        if found:
                            columns_per_clause[k].add(column)

        tokens_queue = [sqlparse.parse(self.query_text)[0], ]
        while len(tokens_queue):
            token = tokens_queue.pop(0)
            if isinstance(token, Comparison) and token.is_group:
                non_empty_tokens = [_ for _ in token if not _.is_whitespace]

                # join predicates
                if isinstance(non_empty_tokens[0], Identifier) and isinstance(non_empty_tokens[2], Identifier):
                    for identifier in [non_empty_tokens[0].value, non_empty_tokens[2].value]:
                        found, column = self.find_column(identifier)
                        if found:
                            columns_per_clause['join'].add(column)

                # filter predicates
                elif isinstance(non_empty_tokens[0], Identifier):
                    identifier = non_empty_tokens[0].value
                    found, column = self.find_column(identifier)
                    if found:
                        columns_per_clause['filter'].add(column)

                elif isinstance(non_empty_tokens[0], Function):
                    identifiers = []
                    tokens_queue_ = [non_empty_tokens[0], ]
                    while len(tokens_queue_):
                        token_ = tokens_queue_.pop(0)
                        if isinstance(token_, Identifier):
                            identifiers.append(token_.value)
                        if token_.is_group:
                            for tok_ in token_:
                                tokens_queue_.append(tok_)

                    for identifier in identifiers:
                        found, column = self.find_column(identifier)
                        if found:
                            columns_per_clause['filter'].add(column)
                else:
                    logging.debug(f'Cannot parse comparison predicate in {self.query_id}: {token}')

            if token.is_group:
                for tok in token:
                    tokens_queue.append(tok)

        self._indexable_columns_per_clause = columns_per_clause
        return columns_per_clause

    def find_column(self, identifier):
        if '.' in identifier:
            for column in self.indexable_columns:
                _table, _column = identifier.split('.')
                if _table in self._tables_aliases:
                    _table = self._tables_aliases[_table]
                if f"{column.table}.{column.name}" == f"{_table}.{_column}":
                    return True, column
        else:
            for column in self.indexable_columns:
                if column.name == identifier:
                    return True, column

        self.logger.debug(f"Identifier {identifier} not found.")
        return False, None


class Workload:
    def __init__(self, benchmark, queries: List[Query]):
        self.benchmark = benchmark
        self.queries = queries

        self.execution_cost = None
        self.estimation_cost = None

    def indexable_columns(self):
        indexable_columns = set()
        for query in self.queries:
            indexable_columns |= set(query.indexable_columns)
        return sorted(list(indexable_columns))

    def potential_indexes(self):
        return sorted([Index([c]) for c in self.indexable_columns()])
