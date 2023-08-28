import os
import sys
import tempfile
import argparse
from glob import glob
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from sklearn.manifold import TSNE
from skimage.transform import resize
import scipy
import torch

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


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray):
    mu_gen, sigma_gen = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(
        np.dot(sigma_gen, sigma_real), disp=False
    )  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

    return float(fid)


def compute_stats(feats: np.ndarray):
    mu = feats.mean(axis=0)  # [d]
    sigma = np.cov(feats, rowvar=False)  # [d, d]

    return mu, sigma


@torch.no_grad()
def compute_our_fvd(
    videos_fake: np.ndarray, videos_real: np.ndarray, device: str = "cpu"
) -> float:
    detector_kwargs = dict(
        rescale=False, resize=False, return_features=True
    )  # Return raw features before the softmax layer.

    # TODO: Use importlib-resources
    with open("data/i3d_torchscript.pt", "rb") as f:
        detector = torch.jit.load(f).eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return compute_fvd(feats_fake, feats_real)


def viz_short_trajectory(batch_size, fake_data_np, save_path):
    fake_data_np = fake_data_np[:batch_size, :16, :, :, :].reshape(-1, 96, 96, 3)

    fig = plt.figure(figsize=(20, 6))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(5, 16),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    for ax, im in zip(grid, fake_data_np):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def viz_long_trajectory(fake_data_np, save_path):
    fake_data_np = fake_data_np[0, :, :, :, :]

    fig = plt.figure(figsize=(20, 6))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(3, 16),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    for ax, im in zip(grid, fake_data_np):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def qualitative_results_img(
    experiment, n_frames, short_video_save_path, long_video_save_path
):
    fake_data = experiment.get_fake_data(
        nz=experiment.nz,
        nc=experiment.ndim_channel,
        img_size=experiment.img_size,
        dataset=experiment.dataloader.dataset,
        video_lengths=[n_frames],
    )

    fake_data_np = fake_data["videos"].permute(0, 2, 3, 4, 1) / 2 + 0.5
    fake_data_np = fake_data_np.detach().cpu().numpy().squeeze()

    viz_short_trajectory(
        experiment.config.experiment.batch_size,
        fake_data_np,
        save_path=short_video_save_path,
    )
    viz_long_trajectory(fake_data_np, save_path=long_video_save_path)


def qualitative_results_latent(experiment, n_frames, batch_size, save_path):
    # embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    # X_transformed = embedding.fit_transform(X, init="pca")

    Z, _, _, _ = experiment.get_latent_sample(
        rnn_type=experiment.config.experiment.architecture,
        batch_size=batch_size,
        d_C=experiment.ndim_content,
        d_E=experiment.ndim_epsilon,
        d_L=experiment.ndim_label,
        d_P=experiment.ndim_physics,
        d_N=experiment.ndim_q2,
        device=experiment.device,
        nz=experiment.nz,
        dataset=experiment.dataloader.dataset,
        rnn=experiment.rnn,
        n_frames=n_frames,
    )

    X = [
        Z[i, 0, : experiment.ndim_q].data.cpu().numpy().squeeze()
        for i in range(batch_size)
    ]
    X = np.asarray(X).reshape(-1, experiment.ndim_q)

    perplexity_values = (2, 5, 30, 50, 100)
    tsnes = [
        TSNE(n_components=2, perplexity=p, init="random").fit_transform(X)
        for p in perplexity_values
    ]

    fig, axs = plt.subplots(
        ncols=len(tsnes), nrows=1, figsize=(20, 6), layout="constrained"
    )

    for i, tsne in enumerate(tsnes):
        axs[i].plot(tsne[:, 0], tsne[:, 1], ".")
        axs[i].axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)


def load_generated_imgs(generated_img_path):
    img_files = glob(generated_img_path + "/*")
    num_files = len(img_files)
    imgs_list = [np.load(img_files[i]) for i in range(num_files)]

    b, s, c, h, w = imgs_list[0].shape

    imgs = np.array(imgs_list)[:, :, :16, :, :]
    imgs = imgs.reshape(num_files * b, s, c, h, w)
    imgs = imgs.transpose(0, 1, 3, 4, 2)

    return imgs


def load_real_imgs(videos_dataloader, fvd_batch_size):
    # TODO: Get fvd_batch_size no. of images
    vd = iter(videos_dataloader)
    data, _, _ = next(vd)
    np_data = data.detach().cpu().numpy().transpose(0, 2, 3, 4, 1)

    return np_data


def fvd_resize(imgs, num_videos, height, width, chan):
    resized_imgs = []

    for vid in imgs[:num_videos]:
        resized_video = np.asarray([resize(img, (height, width, chan)) for img in vid])
        resized_imgs.append(resized_video)

    resized_imgs = np.asarray(resized_imgs).astype(np.float32)

    return resized_imgs


def quantitative_result(
    videos_dataloader,
    generated_img_path,
    fvd_batch_size,
    num_videos,
    height,
    width,
    chan,
):
    imgs = load_generated_imgs(generated_img_path)
    resized_imgs = fvd_resize(imgs, num_videos, height, width, chan)

    np_data = load_real_imgs(videos_dataloader, fvd_batch_size)
    resized_np_data = fvd_resize(np_data, num_videos, height, width, chan)

    # num_videos, video_len, height, width, chan
    our_fvd_result = compute_our_fvd(resized_np_data, resized_imgs)

    return our_fvd_result


def save_generated_images(experiment, generated_img_path, n=1):
    for i in range(n):
        fake_data = experiment.get_fake_data(
            nz=experiment.nz,
            nc=experiment.ndim_channel,
            img_size=experiment.img_size,
            dataset=experiment.dataloader.dataset,
            video_lengths=[50],
        )
        fake_data_np = fake_data["videos"].permute(0, 2, 1, 3, 4).detach().cpu().numpy()
        filename = generated_img_path + str(i).zfill(4)
        np.save(filename, fake_data_np)


def compute_ckpt_fvd(
    experiment, videos_dataloader, fvd_batch_size, num_videos, height, width, chan, n=1
):
    """ """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = tmpdirname + "/"
        save_generated_images(experiment, tmpdirname, n=n)

        fvd = quantitative_result(
            videos_dataloader,
            tmpdirname,
            fvd_batch_size,
            num_videos,
            height,
            width,
            chan,
        )

    return fvd


def main(*args):
    args = get_parser().parse_args(args)
    config = load_config(args.config_path)

    experiment = Experiment(config)
    experiment.load_epoch()  # load latest epoch

    # -------------------------------
    # 1 Generate videos
    # -------------------------------

    qualitative_results_img(
        experiment, 100, "out/short_videos_out.png", "out/long_videos_out.png"
    )

    # -------------------------------
    # 2 Fréchet Video Distance to measure quality of generated videos
    # -------------------------------

    for epoch in experiment.saved_epochs():
        experiment.load_epoch(epoch)
        fvd = compute_ckpt_fvd(
            experiment=experiment,
            videos_dataloader=experiment.dataloader,
            fvd_batch_size=int(2048 / 16),
            num_videos=16,
            height=224,
            width=224,
            chan=3,
        )
        print(epoch, fvd)

    # -------------------------------
    # 3 Show Configuration Space
    # -------------------------------
    qualitative_results_latent(experiment, 2, 1024, "out/config_space.svg")


if __name__ == "__main__":
    main(*sys.argv[1:])
