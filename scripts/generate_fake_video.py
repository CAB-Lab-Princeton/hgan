import os
import sys
import argparse
import logging
from hgan.configuration import load_config
from hgan.experiment import Experiment
from hgan.utils import setup_reproducibility


logger = logging.getLogger(__name__)


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

    setup_reproducibility(config.experiment.seed)

    experiment = Experiment(config)
    # load from local path instead of original output path
    experiment.config.paths.output = os.path.dirname(args.config_path)

    experiment.eval()
    experiment.load_epoch()
    real_data = experiment.get_real_data()
    label_and_props = real_data["label_and_props"][0]

    data = experiment.get_fake_data(n_frames=16, label_and_props=label_and_props)

    # (batch_size, nc, T, img_size, img_size) => (batch_size, T, img_size, img_size, nc)
    videos = data["videos"].permute(0, 2, 3, 4, 1)
    videos = videos.detach().cpu().numpy().squeeze()

    experiment.save_video(folder=".", video=videos[0], filename="generated")


if __name__ == "__main__":
    main(*sys.argv[1:])
