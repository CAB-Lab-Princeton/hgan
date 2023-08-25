import argparse
from hgan.configuration import load_config
from hgan.experiment import Experiment


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration.ini specifying experiment parameters",
    )
    return parser


def main(*args):
    args = get_parser().parse_args(args)

    config = load_config(args.config_path)
    experiment = Experiment(config)
    experiment.train()
