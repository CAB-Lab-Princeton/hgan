import os
import sys
import argparse
import logging
import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from hgan.configuration import load_config
from hgan.experiment import Experiment
from hgan.hgn_datasets import all_systems_hgn, variable_physics_hgn
from hgan.hgn.environments.environment_factory import EnvFactory


logger = logging.getLogger("hgan")
matplotlib.use("agg")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to configuration.ini specifying experiment parameters",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder where results will be generated",
    )
    parser.add_argument(
        "--every-nth",
        type=int,
        default=1,
        help="Process every nth epoch checkpoint encountered (default 1)",
    )
    parser.add_argument(
        "--generated-videos-timeslots",
        type=int,
        default=8,
        help="Number of timeslots to include in generated videos pngs (default 8)",
    )
    parser.add_argument(
        "--generated-videos-samples",
        type=int,
        default=3,
        help="Number of samples to include in generated videos pngs (default 3)",
    )
    parser.add_argument(
        "--latent-batch-size",
        type=int,
        default=1024,
        help="Number of latent samples to generate for TSNE embedding (default 1024)",
    )
    parser.add_argument(
        "--calculate-fvd",
        dest="calculate_fvd",
        action="store_true",
        default=False,
        help="Calculate fvd score for every processed epoch (expensive operation!)",
    )
    parser.add_argument(
        "--fvd-batch-size",
        type=int,
        default=16,
        help="Number of real/fake videos to consider for fvd calculation (default (%default)s)",
    )
    parser.add_argument(
        "--fvd-on-cpu",
        action="store_true",
        default=False,
        help="Whether to run FVD on cpu (for low memory GPUs; default False)",
    )
    parser.add_argument(
        "--system_name",
        type=str,
        default="mass_spring",
        help="The system to run the eval on.",
    )
    parser.add_argument(
        "--override-output-folder",
        action="store_true",
        default=False,
        help="Whether to assume that the output folder is the same as the config path folder.",
    )
    return parser


def qualitative_results_img(
    experiment,
    png_path,
    fake=True,
    timeslots=8,
    samples=3,
    title="",
    epoch=None,
    save_video=False,
    label_and_props=None,
    colors=None,
):
    videos = (
        experiment.get_fake_data(label_and_props=label_and_props, colors=colors)
        if fake
        else experiment.get_real_data()
    )["videos"][
        :samples, ...
    ]  # (samples, nc, T, img_size, img_size)
    videos = videos.detach().cpu().numpy()

    # (samples, nc, T, img_size, img_size) => (samples, T, img_size, img_size, nc)
    videos = videos.transpose(0, 2, 3, 4, 1)

    if save_video:
        experiment.save_video(
            os.path.dirname(png_path),
            videos[0],
            epoch=epoch,
            prefix="fake_" if fake else "real_",
        )

    videos = videos[:samples, :timeslots, :, :, :]
    # Create a ndarray of video frames: timeslots, then samples_per_timeslot
    videos = videos.reshape((-1, *videos.shape[2:]))

    fig = plt.figure(figsize=(20, 6))
    fig.suptitle(title)
    # A grid in which each column represents a timeslot and each row a different instantiation of the
    # video at that time slot
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(samples, timeslots),
        axes_pad=0.1,
    )

    for ax, im in zip(grid, videos):
        ax.axis("off")
        ax.imshow(im)

    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    plt.savefig(png_path)
    plt.close(fig=fig)


def qualitative_results_latent(
    experiment,
    label_and_props,
    png_paths,
    perplexity_values=(2, 5, 30, 50, 100),
    title="",
    n_frames=30,
    projections=("tsne", "pca"),
):
    batch_size = label_and_props.shape[0]
    Z, _, eps_motion = experiment.get_latent_sample(
        batch_size=batch_size, n_frames=1, label_and_props=label_and_props
    )  # shape (batch_size, n_frames, |ndim_q + ndim_p + ndim_content + ndim_label|, 1, 1)

    trajectory, _ = experiment.rnn(
        torch.concat((label_and_props[0], eps_motion[0])).unsqueeze(0),
        n_frames=n_frames,
    )
    trajectory = trajectory[:, 0, : experiment.ndim_q].data.cpu().numpy().squeeze()

    X_train = [
        Z[i, 0, : experiment.ndim_q].data.cpu().numpy().squeeze()
        for i in range(batch_size)
    ]
    X_train = np.asarray(X_train).reshape(
        -1, experiment.ndim_q
    )  # shape (batch_size, ndim_q)
    X_test = trajectory

    size_train = X_train.shape[0]
    X = np.vstack((X_train, X_test))

    for projection, png_path in zip(projections, png_paths):
        if projection == "tsne":
            fig, axs = plt.subplots(
                ncols=len(perplexity_values),
                nrows=1,
                figsize=(20, 6),
                layout="constrained",
            )
            fig.suptitle(title)

            for i, p in enumerate(perplexity_values):
                projection = TSNE(n_components=2, perplexity=p, init="random")
                projected = projection.fit_transform(X)

                projected_train = projected[:size_train]
                projected_test = projected[size_train:]
                axs[i].plot(projected_train[:, 0], projected_train[:, 1], ".")
                axs[i].plot(projected_test[:, 0], projected_test[:, 1], "x")
                axs[i].set_title(f"Perplexity {p}")
                axs[i].axis("off")
        else:
            projection = PCA(n_components=2)
            projected_train = projection.fit_transform(X_train)
            projected_test = projection.transform(X_test)
            fig, ax = plt.subplots(figsize=(20, 6))
            fig.suptitle(title)
            ax.plot(projected_train[:, 0], projected_train[:, 1], ".")
            ax.plot(projected_test[:, 0], projected_test[:, 1], "x")
            ax.axis("off")

        os.makedirs(os.path.dirname(png_path), exist_ok=True)
        plt.savefig(png_path)
        plt.close(fig=fig)


def main(*args):

    device = "cpu" if not torch.cuda.is_available() else None

    args = get_parser().parse_args(args)
    config = load_config(args.config_path)

    output_folder = args.output_folder
    config.save(output_folder)

    experiment = Experiment(config)
    if args.override_output_folder:
        experiment.config.paths.output = os.path.dirname(args.config_path)
    experiment.eval()

    system_index = all_systems_hgn.index(args.system_name)
    system_args = variable_physics_hgn[args.system_name]
    system_args = {
        k: (v() if not isinstance(v, list) else [_v() for _v in v])
        for k, v in system_args.items()
    }
    system = EnvFactory.get_environment(
        experiment.dataloader.dataset.system_name_mapping[args.system_name],
        **system_args,
    )
    props = torch.tensor(system.physical_properties(vec_length=experiment.ndim_physics))

    colors = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1])  # rgbrgbrgb
    colors = (
        colors.unsqueeze(0).repeat(experiment.batch_size, 1).to(experiment.device)
    )  # (batch_size, ndim_color)

    saved_epochs = experiment.saved_epochs()
    for epoch in saved_epochs[:: args.every_nth]:

        logger.info(f"Processing epoch {epoch}")
        experiment.load_epoch(epoch, device=device)

        label_and_props = torch.cat(
            (
                experiment.system_embedding(torch.tensor([system_index])).squeeze(),
                props,
            )
        )

        label_and_props = (
            label_and_props.unsqueeze(0)
            .repeat(experiment.batch_size, 1)
            .to(experiment.device)
        )  # (batch_size, ndim_label + ndim_physics)

        Z, _, _ = experiment.get_latent_sample(
            batch_size=config.experiment.batch_size,
            n_frames=config.video.generator_frames,
            label_and_props=label_and_props,
        )
        Z_motion = Z[0, :, : experiment.ndim_epsilon, :, :].squeeze()
        hnn_input = torch.concat(  # Note order: label_props, then Z_motion
            (
                label_and_props[0]
                .unsqueeze(0)
                .repeat(config.video.generator_frames, 1),
                Z_motion,
            ),
            axis=1,
        )
        energy = experiment.rnn.hnn(hnn_input)
        std_energy = float(torch.std(energy.squeeze()))

        with open(os.path.join(output_folder, "energy.txt"), "a") as f:
            f.write(f"epoch={epoch}, std_energy={std_energy}\n")

        logger.info("  Generating Videos Image")
        qualitative_results_img(
            experiment,
            f"{output_folder}/videos_{epoch:06d}.png",
            timeslots=args.generated_videos_timeslots,
            samples=args.generated_videos_samples,
            title=f"Epoch {epoch}",
            epoch=epoch,
            save_video=True,
            fake=True,
            label_and_props=label_and_props,
            colors=colors,
        )

        logger.info("  Generating Latent Features Image")
        # For the first label_and_props sampled 1024 times from the latent space,
        # plot the TSNE embedding of the q part of the latent space
        qualitative_results_latent(
            experiment=experiment,
            label_and_props=label_and_props[0].unsqueeze(0).repeat(1024, 1),
            png_paths=[
                f"{output_folder}/config_{epoch:06d}_tsne.png",
                f"{output_folder}/config_{epoch:06d}_pca.png",
            ],
            title=f"Epoch {epoch}",
        )

        if args.calculate_fvd:
            logger.info("  Calculating FVD Score")
            fvd_device = "cpu" if args.fvd_on_cpu else experiment.device
            fvd = experiment.fvd(device=fvd_device, max_videos=args.fvd_batch_size)
            fvd_score_file = os.path.join(output_folder, "fvd_scores.txt")
            with open(fvd_score_file, "a") as f:
                f.write(f"epoch={epoch}, fvd={fvd}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
