import os
import glob

import skvideo.io
from skimage.transform import resize
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from hgan.utils import trim_noise
from hgan.configuration import config


class AviDataset(Dataset):
    def __init__(self, datapath, T):
        self.T = T
        self.datapath = os.path.join(datapath, "resized_data")
        self.files = glob.glob(os.path.join(self.datapath, "*"))

        self.videos = self.get_videos()
        self.n_videos = len(self.videos)
        self.video_lengths = [video.shape[1] for video in self.videos]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        start = np.random.randint(0, video.shape[1] - (self.T + 1))
        end = start + self.T
        return video[:, start:end, ...].astype(np.float32)

    def get_videos(self):
        videos = [skvideo.io.vread(file) for file in self.files]
        # transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
        videos = [video.transpose(3, 0, 1, 2) / 255.0 for video in videos]

        return videos


class ToyPhysicsDataset(Dataset):
    def __init__(self, datapath, delta=1, train=True, resize=True, normalize=True):
        train_test = "train" if train else "test"
        self.T = config.video.frames
        self.resize = resize
        self.normalize = normalize
        # self.verbose = verbose

        self.delta = delta
        self.datapath = os.path.join(datapath, train_test)
        self.files = glob.glob(os.path.join(self.datapath, "*.npy"))

        # TODO: Where did these come from?
        self.video_lengths = [
            39,
            112,
            101,
            120,
            43,
            56,
            64,
            68,
            36,
            92,
            36,
            38,
            84,
            111,
            56,
            67,
            119,
            88,
            68,
            57,
            50,
            52,
            42,
            85,
            76,
            77,
            37,
            41,
            48,
            60,
        ]  # [30]
        self.video_lengths = [84] * 30

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        filename = os.path.join(self.datapath, f"{idx:06}.npy")

        vid = np.load(filename)
        # import pdb; pdb.set_trace()
        vid = vid[:: self.delta]  # orig dt is 0.05
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - (self.T + 1))
        end = start + self.T
        vid = vid[start:end]

        vid = (
            np.asarray(
                [resize(img, (config.todo.magic, config.todo.magic, nc)) for img in vid]
            )
            if self.resize
            else vid
        )
        # vid = np.asarray([resize(img, (96, 96, nc)) for img in vid])
        # transpose each video to (nc, n_frames, img_size, img_size), and divide by 255
        vid = vid.transpose(3, 0, 1, 2)
        # normalize -1 1
        vid = (vid - 0.5) / 0.5 if self.normalize else vid
        # vid = (vid - 0.5)/0.5

        return vid.astype(np.float32)


class ToyPhysicsDatasetNPZ(Dataset):
    def __init__(self, datapath, num_frames, delta=1, train=True):
        train_test = "train" if train else "test"
        self.num_frames = num_frames
        self.delta = delta
        self.datapath = os.path.join(datapath, train_test)
        self.files = glob.glob(os.path.join(self.datapath, "*.npz"))

        # TODO: Where do these come from?
        # self.video_lengths = [39, 112, 101, 120, 43, 56, 64, 68, 36, 92, 36, 38, 84, 111, 56, 67, 119, 88, 68, 57, 50, 52, 42, 85, 76, 77, 37, 41, 48, 60]
        self.video_lengths = [200] * 30

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.datapath, str(idx).zfill(5) + ".npz")

        vid = np.load(filename)["arr_0"]
        vid = vid[:: self.delta]  # orig dt is 0.05
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - (self.num_frames + 1))
        end = start + self.num_frames
        vid = vid[start:end]

        if img_size != config.todo.magic:
            vid = np.asarray(
                [resize(img, (config.todo.magic, config.todo.magic, nc)) for img in vid]
            )

        # transpose each video to (nc, n_frames, img_size, img_size), and divide by 255
        vid = vid.transpose(3, 0, 1, 2)

        if config.video.normalize:
            vid = (vid - 0.5) / 0.5

        return vid.astype(np.float32)


def build_dataloader(args):
    videos_dataset = get_dataset(args)
    videos_dataloader = DataLoader(
        videos_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    return videos_dataloader


def get_dataset(args):
    """
    builds a dataloader

    Args:
        args (ArgumentParser): experiment parameters

    Returns:
        Dataset
    """
    return ToyPhysicsDatasetNPZ(args.datapath, num_frames=config.video.frames)


def get_real_data(args, videos_dataloader):
    """
    gets a random sample from the dataset

    Args:
        args (ArgumentParser): experiment parameters
        videos_dataloader (DataLoader): dataloader

    Returns:
        real_data (dict): (sample videos, sample images)
    """

    real_videos = next(iter(videos_dataloader))
    real_videos = real_videos.to(args.device)
    real_videos = Variable(real_videos)

    real_videos_frames = real_videos.shape[2]

    real_img = real_videos[:, :, np.random.randint(0, real_videos_frames), :, :]

    real_data = {"videos": real_videos, "img": real_img}

    return real_data


def get_fake_data(args, video_lengths, rnn, gen_i, T=None):
    """
    gets a random sample from the generator

    Args:
        args (ArgumentParser): experiment parameters
        video_lengths (list): video lenght distribution for avi dataset
        rnn: motion model
        gen_i: image generator

    Returns:
        fake_data (dict): (sample videos, sample images)
    """

    T = config.video.frames if T is None else T

    n_videos = len(video_lengths)
    idx = np.random.randint(0, n_videos)
    n_frames = video_lengths[idx]

    # Z.size() => (batch_size, n_frames, nz, 1, 1)
    Z, dz = get_latent_sample(args, rnn, n_frames)
    # trim => (batch_size, T, nz, 1, 1)
    Z = trim_noise(Z, T=T)
    Z_reshape = Z.contiguous().view(args.batch_size * T, args.nz, 1, 1)

    # generate videos
    fake_videos = gen_i(Z_reshape)

    # import pdb; pdb.set_trace()
    fake_videos = fake_videos.view(
        args.batch_size, T, args.nc, args.img_size, args.img_size
    )
    # transpose => (batch_size, nc, T, img_size, img_size)
    fake_videos = fake_videos.transpose(2, 1)
    # img sampling
    fake_img = fake_videos[:, :, np.random.randint(0, T), :, :]

    fake_data = {"videos": fake_videos, "img": fake_img, "latent": Z, "dlatent": dz}

    return fake_data


def get_latent_sample(args, rnn, n_frames):
    if args.rnn_type in ["gru", "hnn_simple"]:
        return get_simple_sample(args, rnn, n_frames)
    elif args.rnn_type in ["hnn_phase_space"]:
        return get_phase_space_sample(args, rnn, n_frames)
    else:
        return get_mass_sample(args, rnn, n_frames)


def get_random_content_vector(args, n_frames):
    z_C = Variable(torch.randn(args.batch_size, args.d_C))
    #  repeat z_C to (batch_size, n_frames, d_C)
    z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
    z_C = z_C.to(args.device)

    return z_C


def compute_simple_motion_vector(args, n_frames, rnn):
    eps = Variable(torch.randn(args.batch_size, args.d_E))
    eps = eps.to(args.device)

    rnn.initHidden(args.batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M = rnn(eps, n_frames).transpose(1, 0)
    return z_M


def compute_phase_space_motion_vector(args, n_frames, rnn):
    eps = Variable(torch.randn(args.batch_size, args.d_E))
    eps = eps.to(args.device)

    rnn.initHidden(args.batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M, dz_M = rnn(eps, n_frames)
    z_M = z_M.transpose(1, 0)
    dz_M = dz_M.transpose(1, 0)
    return z_M, dz_M


def compute_mass_motion_vector(args, n_frames, rnn):
    eps = Variable(torch.randn(args.batch_size, args.d_E))
    Z_mass = Variable(torch.randn(args.batch_size, args.d_N))

    eps = eps.to(args.device)
    Z_mass = Z_mass.to(args.device)

    rnn.initHidden(args.batch_size)
    # notice that 1st dim of hnn outputs is seq_len, 2nd is batch_size
    z_M, z_mass = rnn(eps, Z_mass, n_frames)

    z_M = z_M.transpose(1, 0)
    z_mass = z_mass.unsqueeze(1).repeat(1, n_frames, 1)

    return z_M, z_mass


def get_simple_sample(args, rnn, n_frames):
    """
    generates latent sample that serves as an input to the generator
    the motion sample is generated with rnn

    Args:
        args (ArgumentParser): experiment parameters
        rnn: motion model
        n_frames (int): video length

    Returns:
        z (Tensor): latent sample
    """

    z_C = get_random_content_vector(args, n_frames)
    z_M = compute_simple_motion_vector(args, n_frames, rnn)
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)

    return z.view(args.batch_size, n_frames, args.nz, 1, 1), None


def get_phase_space_sample(args, rnn, n_frames):
    """
    generates latent sample that serves as an input to the generator
    the motion sample is generated with rnn

    Args:
        args (ArgumentParser): experiment parameters
        rnn: motion model
        n_frames (int): video length

    Returns:
        z (Tensor): latent sample
    """

    z_C = get_random_content_vector(args, n_frames)
    z_M, dz_M = compute_phase_space_motion_vector(args, n_frames, rnn)
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)

    return z.view(args.batch_size, n_frames, args.nz, 1, 1), dz_M


def get_mass_sample(args, rnn, n_frames):
    """
    generates latent sample that serves as an input to the generator
    the motion sample is generated with rnn

    Args:
        args (ArgumentParser): experiment parameters
        rnn: motion model
        n_frames (int): video length

    Returns:
        z (Tensor): latent sample
    """

    z_C = get_random_content_vector(args, n_frames)
    z_M, z_mass = compute_mass_motion_vector(args, n_frames, rnn)
    # import pdb; pdb.set_trace()
    z = torch.cat((z_M, z_mass, z_C), 2)  # z.size() => (batch_size, n_frames, nz)
    return z.view(args.batch_size, n_frames, args.nz, 1, 1), None
