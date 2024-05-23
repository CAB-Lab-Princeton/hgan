import argparse
import logging
import os.path
from hgan.configuration import load_config
from hgan.experiment import Experiment


logger = logging.getLogger("hgan")


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

    logging_file_handler = logging.FileHandler(
        os.path.join(config.paths.output, "hgan.log")
    )
    logging_file_handler.setLevel(logging.NOTSET)
    logger.addHandler(logging_file_handler)

    experiment = Experiment(config)
    experiment.train()
