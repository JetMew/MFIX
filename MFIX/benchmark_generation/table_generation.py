import os
import pdb
import re
import subprocess

from MFIX.dbconnector.base import DBConnector
from MFIX.index import Column, Table
from MFIX.utils.conversion import b_to_mb
from MFIX.utils.logging_utils import get_logger

# TODO: modify path
TPCH_PATH = """PATH_TO/tpch-kit/dbgen"""
TPCDS_PATH = """PATH_TO/tpcds-kit/tools"""
JOB_PATH = """PATH_TO/imdb"""
IGNORE_PK = False


class TableGenerator:
    def __init__(
            self,
            benchmark_name,
            scale_factor,
            db_connector: DBConnector,
            explicit_database_name=None
    ):
        self.logger = get_logger(self.__class__.__name__)

        self.scale_factor = scale_factor
        self.benchmark_name = benchmark_name
        self.db_connector = db_connector
        self.explicit_database_name = explicit_database_name
        self.database_name = self.generate_database_name()

        if self.benchmark_name == 'job':
            self.directory = JOB_PATH
        elif self.benchmark_name == "tpch":
            self.directory = TPCH_PATH
        elif self.benchmark_name == "tpcds":
            self.directory = TPCDS_PATH
        else:
            raise NotImplementedError("only tpch/tpcds implemented.")

        self.tables = []
        self.columns = []
        self._prepare()
        if self.database_name not in self.db_connector.get_databases():
            self._generate_data()
            self._create_database()
        else:
            self.logger.debug("Database with the given scale factor already exists.")
        self._read_column_names()
        self._get_table_row_count()

    def generate_database_name(self):
        if self.explicit_database_name:
            return self.explicit_database_name
        else:
            return f"{self.benchmark_name}__{self.scale_factor}".replace('.', '_')

    def _prepare(self):
        if self.benchmark_name == 'job':
            self.create_table_statements_file = 'schema.sql'
        elif self.benchmark_name == "tpch":
            if self.db_connector.dbtype == 'postgres':
                self.make_command = ["make", "DATABASE=POSTGRESQL"]
            else:
                raise NotImplementedError("Auto benchmark generation for MySQL not implemented.")
            self.create_table_statements_file = "dss.ddl"
            self.cmd = ["./dbgen", "-s", str(self.scale_factor), "-f"]

        elif self.benchmark_name == "tpcds":
            self.make_command = ["make"]
            self.create_table_statements_file = "tpcds.sql"
            self.cmd = ["./dsdgen", "-SCALE", str(self.scale_factor), "-FORCE"]
        else:
            raise NotImplementedError("only tpch/ds implemented.")

    def _run_make(self):
        if "dbgen" not in self._files() and "dsdgen" not in self._files():
            self.logger.info("Running make in {}".format(self.directory))
            self._run_command(self.make_command)
        else:
            self.logger.info("No need to run make")

    def _run_command(self, command):
        cmd_out = "[SUBPROCESS OUTPUT] "
        p = subprocess.Popen(
            command,
            cwd=self.directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with p.stdout:
            for line in p.stdout:
                self.logger.info(cmd_out + line.decode("utf-8").replace("\n", ""))
        p.wait()

    def _files(self):
        return os.listdir(self.directory)

    def _table_files(self):
        self.table_files = [x for x in self._files() if ".tbl" in x or ".dat" in x]

    def _generate_data(self):
        if self.benchmark_name == 'job':
            return

        self.logger.info("Generating {} data".format(self.benchmark_name))
        self.logger.info("scale factor: {}".format(self.scale_factor))
        self._run_make()
        self._run_command(self.cmd)
        if self.benchmark_name == "tpcds":
            current_path = os.path.abspath(__file__)
            script_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(current_path))), "scripts/replace_in_dat.sh"
            )
            self._run_command(["bash", script_path])
        self.logger.info("[Generate command] " + " ".join(self.cmd))
        self._table_files()
        assert len(self.table_files)
        self.logger.info("Files generated: {}".format(self.table_files))

    def _create_database(self):
        self.db_connector.create_database(self.database_name)
        filename = self.directory + "/" + self.create_table_statements_file
        with open(filename, "r") as file:
            create_statements = file.read()
        if IGNORE_PK:
            # Do not create primary keys
            create_statements = re.sub(r",\s*primary key (.*)", "", create_statements)
        self.db_connector.dbname = self.database_name
        self.db_connector.create_connection()
        self._create_tables(create_statements)
        self._load_table_data()

    def _create_tables(self, create_statements):
        self.logger.info("Creating tables")
        for create_statement in create_statements.split(";")[:-1]:
            self.db_connector.execute(create_statement)
        self.db_connector.commit()

    def _load_table_data(self):
        self.logger.info("Loading data into the tables")
        if self.benchmark_name == 'job':
            filename = self.directory + "/" + 'load_data.sql'
            with open(filename, "r") as file:
                load_statements = file.read()
            for load_statement in load_statements.split(";")[:-1]:
                self.db_connector.execute(load_statement)
            self.db_connector.commit()
            self.db_connector.analyze()
            return

        for filename in self.table_files:
            table = filename.replace(".tbl", "").replace(".dat", "")
            path = os.path.join(self.directory, filename)
            size = os.path.getsize(path)
            size_string = f"{b_to_mb(size):,.4f} MB"
            self.logger.info(f"Loading file {filename}: {size_string}")
            self.db_connector.import_data(table, path)
            os.remove(os.path.join(self.directory, filename))
        self.db_connector.commit()

    def _read_column_names(self):
        # Read table and column names from 'create table' statements
        filename = os.path.join(self.directory, self.create_table_statements_file)
        with open(filename, "r") as file:
            data = file.read().lower()
        create_tables = data.split("create table ")[1:]
        for create_table in create_tables:
            splitted = create_table.split("(", 1)
            table = Table(splitted[0].strip())
            self.tables.append(table)
            for column in splitted[1].split(",\n"):
                name = column.lstrip().split(" ", 1)[0]
                if name == "primary":
                    continue
                column_object = Column(name)
                table.add_column(column_object)
                self.columns.append(column_object)

        self.tables.sort()
        self.columns.sort()

    def _get_table_row_count(self):
        self.db_connector.dbname = self.database_name
        self.db_connector.create_connection()

        for table in self.tables:
            table.row_count = int(self.db_connector.get_table_row_count(table.name))
