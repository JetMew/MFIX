import argparse
from MFIX.utils.config_parser import parse_args
from MFIX.index_selection import IndexSelection


if __name__ == '__main__':
    args_base, args_algo = parse_args(config_file=config_file)
    adv = IndexSelection(args_algo, **args_base)
    adv.run_algorithms()
