from functools import total_ordering


class Column:
    def __init__(self, name):
        self.name = name.lower()
        self.table = None  # set when add column to corresponding table
        self.selectivity = None

    def __lt__(self, other):
        if self.table == other.table:
            return self.name < other.name
        return self.table < other.table

    def __repr__(self):
        return f"C {self.table}.{self.name}"

    # We cannot check self.table == other.table here since Table.__eq__()
    # internally checks Column.__eq__. This would lead to endless recursions.
    def __eq__(self, other):
        if not isinstance(other, Column):
            return False

        assert (
            self.table is not None and other.table is not None
        ), "Table objects should not be None for Column.__eq__()"

        return self.table.name == other.table.name and self.name == other.name

    def __hash__(self):
        return hash((self.name, self.table.name))


class Table:
    def __init__(self, name):
        self.name = name.lower()
        self.columns = []
        self.row_count = None

    def add_column(self, column):
        column.table = self
        self.columns.append(column)

    def add_columns(self, columns):
        for column in columns:
            self.add_column(column)

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        if not isinstance(other, Table):
            return False

        return self.name == other.name and tuple(self.columns) == tuple(other.columns)

    def __hash__(self):
        return hash((self.name, tuple(self.columns)))


@total_ordering
class Index:
    def __init__(self, columns, estimated_size=None, size=None):
        if len(columns) == 0:
            raise ValueError("Index needs at least 1 column")
        self.columns = tuple(columns)
        self.estimated_size = estimated_size
        self.size = size
        self.hypopg_name = None

    # Used to sort indexes
    def __lt__(self, other):
        if len(self.columns) != len(other.columns):
            return len(self.columns) < len(other.columns)
        return self.columns < other.columns

    def __repr__(self):
        columns_string = ",".join(map(str, self.columns))
        return f"I({columns_string})"

    def __eq__(self, other):
        if not isinstance(other, Index):
            return False
        return self.columns == other.columns

    def __hash__(self):
        return hash(self.columns)

    def _column_names(self):
        return [c.name for c in self.columns]

    def is_single_column(self):
        return True if len(self.columns) == 1 else False

    def table(self):
        assert (
            self.columns[0].table is not None
        ), "Table should not be None to avoid false positive comparisions"
        return self.columns[0].table

    def index_idx(self):
        columns = "_".join(self._column_names())
        return f"{self.table()}__{columns}_idx"

    def joined_column_names(self):
        return ",".join(self._column_names())

    def appendable_by(self, other):
        if not isinstance(other, Index):
            return False
        if self.table() != other.table():
            return False
        if not other.is_single_column():
            return False
        if other.columns[0] in self.columns:
            return False
        return True

    def subsumes(self, other):  # check if prefix
        if not isinstance(other, Index):
            return False
        return self.columns[: len(other.columns)] == other.columns

    def prefixes(self):
        index_prefixes = []
        for prefix_width in range(len(self.columns) - 1, 0, -1):
            index_prefixes.append(Index(self.columns[:prefix_width]))
        return index_prefixes


def index_merge(index_1, index_2):
    assert index_1.table() == index_2.table()
    merged_columns = list(index_1.columns)
    for column in index_2.columns:
        if column not in index_1.columns:
            merged_columns.append(column)
    return Index(merged_columns)
