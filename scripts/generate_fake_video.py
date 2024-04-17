import os
import sys
import argparse
import logging
import torch
from hgan.configuration import load_config
from hgan.experiment import Experiment
from hgan.utils import setup_reproducibility
from hgan.hgn_datasets import all_systems_hgn

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

    # ---------- Customize --------- #
    system = "mass_spring"
    system_index = all_systems_hgn.index(system)
    label_and_props = torch.cat(
        (
            experiment.system_embedding(torch.tensor([system_index])).squeeze(),
            torch.tensor([0.5, 2.0, 0, 0, 0, 0, 0, 0, 0, 0]),
        )
    )
    colors = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0])  # red
    # ---------- Customize --------- #

    label_and_props = (
        label_and_props.unsqueeze(0)
        .repeat(experiment.batch_size, 1)
        .to(experiment.device)
    )  # (batch_size, ndim_label + ndim_physics)
    colors = (
        colors.unsqueeze(0).repeat(experiment.batch_size, 1).to(experiment.device)
    )  # (batch_size, ndim_color)
    data = experiment.get_fake_data(
        n_frames=16, label_and_props=label_and_props, colors=colors
    )

    # (1, nc, T, img_size, img_size) => (1, T, img_size, img_size, nc)
    videos = data["videos"].permute(0, 2, 3, 4, 1)
    video = videos.detach().cpu().numpy()[0]

    experiment.save_video(folder=".", video=video, filename="generated")


if __name__ == "__main__":
    main(*sys.argv[1:])
