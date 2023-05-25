import os.path
import torch
import argparse
from hgan.train import run_experiment
from hgan.configuration import config
from hgan.models import GRU, HNNSimple, HNNPhaseSpace, HNNMass

RNN_TYPE = {
    "gru": GRU,
    "hnn_simple": HNNSimple,
    "hnn_phase_space": HNNPhaseSpace,
    "hnn_mass": HNNMass,
}


def get_parser(parser):
    parser.add_argument(
        "--gpuid",
        type=int,
        default=0,
        help="The default GPU ID to use. Set -1 to use cpu.",
    )
    parser.add_argument("--ngpu", type=int, default=1, help="Number of gpus to use")

    parser.add_argument(
        "--video_type",
        type=str,
        nargs="+",
        default=config.experiment.dataset,
        help="dataset choices",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=config.paths.input,
        help="set data directory",
        required=True,
    )
    parser.add_argument(
        "--generated_videos_dir", type=str, default=config.paths.output, required=True
    )
    parser.add_argument(
        "--trained_models_dir", type=str, default=config.paths.output, required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=config.experiment.batch_size,
        help="set batch_size, default: 16",
    )
    parser.add_argument(
        "--niter",
        type=int,
        default=config.experiment.n_epoch,
        help="set num of iterations, default: 120000",
    )
    parser.add_argument(
        "--cyclic_coord_loss",
        type=float,
        default=config.experiment.cyclic_coord_loss,
        help="regularizer for cyclic coord loss",
    )
    parser.add_argument(
        "--rnn_type",
        type=str,
        default=config.experiment.architecture,
        help="select recurrent module",
    )

    parser.add_argument(
        "--print",
        type=int,
        default=config.experiment.print_every,
        help="set num of iterations, for print",
    )
    parser.add_argument(
        "--save_output",
        type=int,
        default=config.experiment.save_video_every,
        help="set num of iterations, for save video",
    )
    parser.add_argument(
        "--save_model",
        type=int,
        default=config.experiment.save_model_every,
        help="set num of iterations, for save model",
    )

    parser.add_argument(
        "--seed", type=int, default=config.experiment.seed, help="set random seed"
    )

    return parser


def add_dependent_args(args, video_type, rnn_type):
    # Add arguments not specified on the command line, but derived from existing arguments
    args.datapath = os.path.join(args.data_dir, video_type)

    # set rnn module {gru, hnn_simple}
    args.rnn = RNN_TYPE[rnn_type]

    if args.gpuid < 0 or not torch.cuda.is_available():
        args.device = "cpu"
    else:
        args.device = f"cuda:{args.gpuid}"

    args.criterion = torch.nn.BCELoss()
    args.criterion.to(args.device)

    args.label = torch.FloatTensor()
    args.label = args.label.to(args.device)

    args.img_size = config.todo.magic
    args.nc = 3
    args.ndf = 32  # from dcgan
    args.ngf = 32

    args.hidden_size = 100  # guess

    # original hyperparameters
    args.d_E = 20
    args.d_C = 50
    args.d_M = args.d_E

    args.lr = 0.0002
    args.betas = (0.5, 0.999)

    # new hyperparameters
    args.q_size = int(args.d_M / 2)
    args.d_N = int(args.q_size**2)

    args.nz = (
        args.q_size + int((args.d_N + args.q_size) / 2) + args.d_C
        if rnn_type == "hnn_mass"
        else args.d_C + args.d_M
    )


def main(*args):
    parser = argparse.ArgumentParser(description=__doc__)
    args = get_parser(parser).parse_args(args)

    video_types = args.video_type
    if isinstance(video_types, str):
        video_types = [video_types]

    rnn_types = args.rnn_type
    if isinstance(rnn_types, str):
        rnn_types = [rnn_types]

    for video_type in video_types:
        for rnn_type in rnn_types:
            add_dependent_args(args, video_type, rnn_type)
            run_experiment(args)
