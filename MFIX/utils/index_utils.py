from MFIX.workload import Workload
from MFIX.utils.logging_utils import get_logger


logger = get_logger('IndexUtils')


def indexes_by_table(indexes):
    indexes_by_table = {}
    for index in indexes:
        table = index.table()
        if table not in indexes_by_table:
            indexes_by_table[table] = []

        indexes_by_table[table].append(index)

    return indexes_by_table


def get_utilized_indexes(
    workload, indexes_per_query, cost_evaluation, detailed_query_information=False
):
    utilized_indexes_workload = set()
    query_details = {}
    for query, indexes in zip(workload.queries, indexes_per_query):
        (
            utilized_indexes_query,
            cost_with_indexes,
        ) = cost_evaluation.which_indexes_utilized_and_cost(Workload(workload.benchmark, [query]), indexes)
        utilized_indexes_workload |= utilized_indexes_query

        logger.debug(
            f'Q{query.query_id}: {len(utilized_indexes_query)} utilized indexes.' +
            ''.join([f'\n\t{index}' for index in utilized_indexes_query])
        )

        if detailed_query_information:
            cost_without_indexes, _, _ = cost_evaluation.calculate_cost(
                Workload(workload.benchmark, [query]), indexes=[]
            )

            query_details[query] = {
                "cost_without_indexes": cost_without_indexes,
                "cost_with_indexes": cost_with_indexes,
                "utilized_indexes": utilized_indexes_query,
            }

    return utilized_indexes_workload, query_details
