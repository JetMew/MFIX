# MFIX

This is the source code to the paper "MFIX: An Efficient and Reliable Index Advisor via Multi-Fidelity Bayesian Optimization". Please refer to the paper for the experimental details.

## Table of Content
* Requirements
* Benchmark Preparation
* Using MFIX

## Requirements
1. Preparations: Python == 3.7

2. Install packages

   ```shell
   pip install -r requirements.txt
   ```

## Benchmark Preparation

### Join-Order-Benchmark (JOB)

Download IMDB Data Set from http://homepages.cwi.nl/~boncz/job/imdb.tgz.

Follow the instructions to load data:
 * For PostgreSQL: https://github.com/gregrahn/join-order-benchmark;

### TPC-H
Download TPC-H benchmark kit from https://github.com/gregrahn/tpch-kit.

### TPC-DS
Download TPC-DS benchmark kit from https://github.com/gregrahn/tpchds-kit.

## Using MFIX
Please modify the user-specified parameters in config.ini before the experiments.
* To specify the task, index, and database connection information, please modify the following parameters:
```ini
[base]
# task related
task_id = pg_job
log_dir = logs/
history_dir = history/
eval_timeout = 90

# index related
index_storage_budget = 6
index_no_redundant = False
index_single_only = False

# database related
dbtype = postgres
host = 127.0.0.1
port = 5432
user = root
passwd =
sock =
dbname = imdbload
```

* To specify the workload information ,please modify the following parameters:
```ini
# workload related
benchmark = job
scale_factor = 1
queries = ['1a', '1b', '1c', '1d', '2a', '2b', '2c', '2d', '3a', '3b', '3c', '4a', '4b', '4c', '5a', '5b', '5c', '6a', '6b', '6c', '6d', '6e', '6f', '7a', '7b', '7c', '8a', '8b', '8c', '8d', '9a', '9b', '9c', '9d', '10a', '10b', '10c', '11a', '11b', '11c', '11d', '12a', '12b', '12c', '13a', '13b', '13c', '13d', '14a', '14b', '14c', '15a', '15b', '15c', '15d', '16a', '16b', '16c', '16d', '17a', '17b', '17c', '17d', '17e', '17f', '18a', '18b', '18c', '19a', '19b', '19c', '19d', '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b', '22c', '22d', '23a', '23b', '23c', '24a', '24b', '25a', '25b', '25c', '26a', '26b', '26c', '27a', '27b', '27c', '28a', '28b', '28c', '29a', '29b', '29c', '30a', '30b', '30c', '31a', '31b', '31c', '32a', '32b', '33a', '33b', '33c']
pickle_workload = True 
```

Set the path to benchmark toolkit in `MFIX/benchmark_generation/table_generation.py`
```python
TPCH_PATH = """PATH_TO/tpch-kit/dbgen"""
TPCDS_PATH = """PATH_TO/tpcds-kit/tools"""
JOB_PATH = """PATH_TO/imdb"""
```

Start MFIX:
```shell
python main.py
```
