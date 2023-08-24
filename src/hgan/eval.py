import os
import re
import tempfile
from glob import glob
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from sklearn.manifold import TSNE
from skimage.transform import resize
import scipy
import torch

from hgan.utils import setup_reproducibility
from hgan.dataset import build_dataloader, get_fake_data, get_latent_sample
from hgan.logger import load, load_ckpt
from hgan.run import get_parser, add_dependent_args
from hgan.configuration import load_config


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

    with open("i3d_torchscript.pt", "rb") as f:
        detector = torch.jit.load(f).eval().to(device)

    videos_fake = torch.from_numpy(videos_fake).permute(0, 4, 1, 2, 3).to(device)
    videos_real = torch.from_numpy(videos_real).permute(0, 4, 1, 2, 3).to(device)

    feats_fake = detector(videos_fake, **detector_kwargs).cpu().numpy()
    feats_real = detector(videos_real, **detector_kwargs).cpu().numpy()

    return compute_fvd(feats_fake, feats_real)


def viz_short_trajectory(args, fake_data_np, save_path=None):
    """ """
    save = save_path is not None

    fake_data_np = fake_data_np[: args.batch_size, :16, :, :, :].reshape(-1, 96, 96, 3)

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

    if save:
        plt.savefig(save_path, transparent=True)


def viz_long_trajectory(args, fake_data_np, save_path=None):
    save = False if save_path is None else True

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

    if save:
        plt.savefig(save_path, transparent=True)


def qualitative_results_img(
    args,
    dataset,
    models,
    optims,
    T,
    short_video_save_path,
    long_video_save_path,
    verbose=False,
):
    """ """
    # generate synthetic videos
    fake_data = get_fake_data(
        rnn_type=args.rnn_type,
        batch_size=args.batch_size,
        d_C=args.d_C,
        d_E=args.d_E,
        d_L=args.d_L,
        d_P=args.d_P,
        d_N=args.d_N,
        device=args.device,
        nz=args.nz,
        nc=args.nc,
        img_size=args.img_size,
        dataset=dataset,
        video_lengths=[100],
        rnn=models["RNN"],
        gen_i=models["Gi"],
        T=T,
    )
    if verbose:
        fake_data["videos"][0].shape

    # convert from torch tensors to numpy arrays
    fake_data_np = fake_data["videos"].permute(0, 2, 3, 4, 1) / 2 + 0.5
    fake_data_np = fake_data_np.detach().cpu().numpy().squeeze()
    if verbose:
        print(fake_data_np.shape, fake_data_np.max(), fake_data_np.min())

    # save sort and long image trajectories
    viz_short_trajectory(args, fake_data_np, save_path=short_video_save_path)
    viz_long_trajectory(args, fake_data_np, save_path=long_video_save_path)


def qualitative_results_latent(args, dataset, models, n_frames, batch_size, save_path):
    # embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0)
    # X_transformed = embedding.fit_transform(X, init="pca")

    tmp_batch_size = args.batch_size
    args.batch_size = batch_size
    Z, dz, _, _ = get_latent_sample(
        rnn_type=args.rnn_type,
        batch_size=args.batch_size,
        d_C=args.d_C,
        d_E=args.d_E,
        d_L=args.d_L,
        d_P=args.d_P,
        d_N=args.d_N,
        device=args.device,
        nz=args.nz,
        dataset=dataset,
        rnn=models["RNN"],
        n_frames=n_frames,
    )

    X = [
        Z[i, 0, : args.q_size].data.cpu().numpy().squeeze()
        for i in range(args.batch_size)
    ]
    X = np.asarray(X).reshape(-1, args.q_size)

    # 2, 5, 30, 50, 100
    X2 = TSNE(n_components=2, perplexity=2, init="random").fit_transform(X)
    X5 = TSNE(n_components=2, perplexity=5, init="random").fit_transform(X)
    X30 = TSNE(n_components=2, perplexity=30, init="random").fit_transform(X)
    X50 = TSNE(n_components=2, perplexity=50, init="random").fit_transform(X)
    X100 = TSNE(n_components=2, perplexity=100, init="random").fit_transform(X)

    fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(20, 6), layout="constrained")

    for ind, X in zip(range(5), [X2, X5, X30, X50, X100]):
        axs[ind].plot(X[:, 0], X[:, 1], ".")
        axs[ind].axis("off")

    plt.savefig(save_path, transparent=True)

    args.batch_size = tmp_batch_size


def load_generated_imgs(generated_img_path):
    """ """
    img_files = glob(generated_img_path + "/*")
    num_files = len(img_files)
    imgs_list = [np.load(img_files[i]) for i in range(num_files)]

    b, s, c, h, w = imgs_list[0].shape

    imgs = np.array(imgs_list)[:, :, :16, :, :]
    imgs = imgs.reshape(num_files * b, s, c, h, w)
    imgs = imgs.transpose(0, 1, 3, 4, 2)

    return imgs


def load_real_imgs(arg, videos_dataloader, fvd_batch_size):
    tmp_batch_size = args.batch_size

    args.batch_size = fvd_batch_size

    vd = iter(videos_dataloader)
    data, _, _ = next(vd)
    np_data = data.detach().cpu().numpy().transpose(0, 2, 3, 4, 1)

    args.batch_size = tmp_batch_size

    return np_data


def fvd_resize(imgs, num_videos, height, width, chan):
    """ """
    resized_imgs = []

    for vid in imgs[:num_videos]:
        resized_video = np.asarray([resize(img, (height, width, chan)) for img in vid])
        resized_imgs.append(resized_video)

    resized_imgs = np.asarray(resized_imgs).astype(np.float32)

    return resized_imgs


def quantitative_result(
    args,
    videos_dataloader,
    generated_img_path,
    fvd_batch_size,
    num_videos,
    height,
    width,
    chan,
    verbose=False,
):
    """ """
    imgs = load_generated_imgs(generated_img_path)
    resized_imgs = fvd_resize(imgs, num_videos, height, width, chan)

    np_data = load_real_imgs(args, videos_dataloader, fvd_batch_size)
    resized_np_data = fvd_resize(np_data, num_videos, height, width, chan)

    # num_videos, video_len, height, width, chan
    our_fvd_result = compute_our_fvd(resized_np_data, resized_imgs)

    return our_fvd_result


def save_generated_images(args, dataset, models, n, T, batch_size, generated_img_path):
    for i in range(n):
        fake_data = get_fake_data(
            rnn_type=args.rnn_type,
            batch_size=batch_size,
            d_C=args.d_C,
            d_E=args.d_E,
            d_L=args.d_L,
            d_P=args.d_P,
            d_N=args.d_N,
            device=args.device,
            nz=args.nz,
            nc=args.nc,
            img_size=args.img_size,
            dataset=dataset,
            video_lengths=[50],
            rnn=models["RNN"],
            gen_i=models["Gi"],
            T=T,
        )
        fake_data_np = fake_data["videos"].permute(0, 2, 1, 3, 4).detach().cpu().numpy()
        filename = generated_img_path + str(i).zfill(4)
        np.save(filename, fake_data_np)


def compute_ckpt_fvd(
    args,
    models,
    videos_dataloader,
    n,
    T,
    batch_size,
    fvd_batch_size,
    num_videos,
    height,
    width,
    chan,
    verbose,
):
    """ """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = tmpdirname + "/"
        save_generated_images(
            args, videos_dataloader.dataset, models, n, T, batch_size, tmpdirname
        )

        fvd = quantitative_result(
            args,
            videos_dataloader,
            tmpdirname,
            fvd_batch_size,
            num_videos,
            height,
            width,
            chan,
            verbose=verbose,
        )

    return fvd


def get_ckpt_list(args):
    ckpt_filenames = glob(
        os.path.join(args.trained_models_dir, "*Discriminator_I_epoch*.model")
    )
    saved_epochs = []
    for f in ckpt_filenames:
        saved_epochs.append(int(re.sub(r"\D", "", f.split("/")[-1])))

    max_saved_epoch = 0 if len(saved_epochs) == 0 else max(saved_epochs)

    return max_saved_epoch, saved_epochs


def select_ckpts(saved_epochs, num_elements=10):
    sorted_saved_epochs = sorted(saved_epochs)
    idx = np.linspace(0, len(sorted_saved_epochs) - 1, num_elements, dtype="int")
    sorted_saved_epochs = np.asarray(sorted_saved_epochs)[idx]
    return sorted_saved_epochs


if __name__ == "__main__":

    config = load_config("/parent/folder/of/configuration.ini/")

    parser = get_parser()
    args = get_parser().parse_args()
    add_dependent_args(args)

    setup_reproducibility(args.seed)
    dataloader = build_dataloader(
        video_type=args.video_type, datapath=args.datapath, batch_size=args.batch_size
    )

    models, optims, max_saved_epoch = load(
        models=None,
        optim=None,
        trained_models_dir=args.trained_models_dir,
        retrain=False,
    )

    # -------------------------------
    # 1 Generate videos
    # -------------------------------

    T = 50

    qualitative_results_img(
        args,
        dataloader.dataset,
        models,
        optims,
        T,
        "short_videos_out.png",
        "long_videos_out.png",
        verbose=True,
    )

    # -------------------------------
    # 2 Fréchet Video Distance to measure quality of generated videos
    # -------------------------------

    height = 224
    width = 224
    chan = 3
    num_videos = 64
    fvd_batch_size = 64

    with open("checkpoint_fvd.csv", "w") as fvd_file:
        header_string = "rnn type, video type, epoch, fvd\n"
        fvd_file.write(header_string)
        max_saved_epoch, saved_epochs = get_ckpt_list(args)

        if max_saved_epoch > 0:
            sorted_saved_epochs = select_ckpts(saved_epochs, num_elements=10)

            T = 16
            n = int(2048 / 16)
            batch_size = 16
            for epoch in sorted_saved_epochs:
                models, optims = load_ckpt(
                    models, optims, epoch, args.trained_models_dir
                )

                fvd = compute_ckpt_fvd(
                    args,
                    models,
                    dataloader,
                    n,
                    T,
                    batch_size,
                    fvd_batch_size,
                    num_videos,
                    height,
                    width,
                    chan,
                    True,
                )

                data_string = f"<rnn_type>, <video_type>, {epoch}, {fvd}\n"
                fvd_file.write(data_string)

    # -------------------------------
    # 3 Show Configuration Space
    # -------------------------------
    config_space_img_save_path = "config_space.svg"

    qualitative_results_latent(
        args, dataloader.dataset, models, 2, 1024, config_space_img_save_path
    )
