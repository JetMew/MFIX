import pickle
import itertools
from MFIX.index import Index
from MFIX.utils.logging_utils import get_logger

logger = get_logger('Candidate')


def candidates_per_query(
        workload,
        max_index_width,
        candidate_generator,
        include_index=True,
        db_conn=None,
        workload_pickle_file=None
):
    candidates = []

    parse_workload = False
    if candidate_generator == syntactically_relevant_indexes_per_clause \
            and workload.queries[0].indexable_columns_per_clause() is None:
        parse_workload = True

    for query in workload.queries:
        candidates.append(
            candidate_generator(
                query=query,
                max_index_width=max_index_width,
                include_index=include_index,
                db_conn=db_conn)
        )

    if parse_workload and workload_pickle_file is not None:
        pickle.dump(workload, open(workload_pickle_file, "wb"))
        logger.info(f"save workload to {workload_pickle_file} with parsed columns.")

    return candidates


def syntactically_relevant_indexes(query, max_index_width, **kwargs):
    # "SAEFIS" or "BFI" see paper linked in DB2Advis algorithm
    # This implementation is "BFI" and uses all syntactically relevant indexes.
    columns = query.indexable_columns
    logger.debug(f"{query}")
    logger.debug(f"Indexable columns: {len(columns)}")

    indexable_columns_per_table = {}
    for column in columns:
        if column.table not in indexable_columns_per_table:
            indexable_columns_per_table[column.table] = set()
        indexable_columns_per_table[column.table].add(column)

    possible_column_combinations = set()
    for table in indexable_columns_per_table:
        columns = indexable_columns_per_table[table]
        for index_length in range(1, max_index_width + 1):
            possible_column_combinations |= set(
                itertools.permutations(columns, index_length)
            )

    logger.debug(f"Potential indexes: {len(possible_column_combinations)}")
    return [Index(p) for p in possible_column_combinations]


def syntactically_relevant_indexes_per_clause(
        query, max_index_width=2, small_table_ignore=True, small_table_threshold=100, include_index=True, db_conn=None
):
    columns_dict = query.get_indexable_columns_per_clause()
    payloads, predicates, group_bys, order_bys = {}, {}, {}, {}

    for column in set(columns_dict['filter'] | columns_dict['join']):
        if column.table not in predicates:
            predicates[column.table] = set()
        predicates[column.table].add(column)

        if db_conn and not column.selectivity:
            column.selectivity = db_conn.get_column_selectivity(column) / column.table.row_count

    for column in columns_dict['select']:
        if column.table not in payloads:
            payloads[column.table] = set()
        payloads[column.table].add(column)

        if db_conn and not column.selectivity:
            column.selectivity = db_conn.get_column_selectivity(column) / column.table.row_count

    for column in columns_dict['group_by']:
        if column.table not in group_bys:
            group_bys[column.table] = set()
        group_bys[column.table].add(column)

        if db_conn and not column.selectivity:
            column.selectivity = db_conn.get_column_selectivity(column) / column.table.row_count

    for column in columns_dict['order_by']:
        if column.table not in order_bys:
            order_bys[column.table] = set()
        order_bys[column.table].add(column)

        if db_conn and not column.selectivity:
            column.selectivity = db_conn.get_column_selectivity(column) / column.table.row_count

    possible_column_combinations = []

    for table, table_predicates in predicates.items():
        if small_table_ignore and table.row_count < small_table_threshold:
            # logger.info(f"{query.query_id}: ignore small table {table}, {table.row_count} rows.")
            continue

        combinations_from_predicates = []
        for j in range(1, min(len(table_predicates), max_index_width) + 1):
            # version 1: permutation
            # combinations_from_predicates += list(itertools.permutations(table_predicates, j))

            # version 2: combination
            combinations = itertools.combinations(table_predicates, j)
            for combination in combinations:
                columns = sorted(combination, key=lambda x: x.selectivity)
                combinations_from_predicates.append(columns)

        possible_column_combinations += combinations_from_predicates

        if include_index:
            combinations_from_payloads = []
            if table in payloads:
                for _ in combinations_from_predicates:
                    includes = payloads[table] - set(_)
                    if len(includes) > 0:
                        includes = sorted(list(includes), key=lambda x: x.selectivity)
                        combinations_from_payloads.append(list(_) + list(includes))

            possible_column_combinations += combinations_from_payloads

    for table, table_order_bys in order_bys.items():
        if small_table_ignore and table.row_count < small_table_threshold:
            continue

        columns = sorted(list(table_order_bys), key=lambda x: x.selectivity)
        possible_column_combinations.append(columns)

    for table, table_group_bys in group_bys.items():
        if small_table_ignore and table.row_count < small_table_threshold:
            continue

        columns = sorted(list(table_group_bys), key=lambda x: x.selectivity)
        possible_column_combinations.append(columns)

    return [Index(p) for p in possible_column_combinations]
