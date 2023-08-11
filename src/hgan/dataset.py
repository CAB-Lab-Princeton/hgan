import os
import glob
import skvideo.io
from skimage.transform import resize
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import jax
import functools
from hgan.dm_hamiltonian_dynamics_suite import datasets
from hgan.utils import trim_noise
from hgan.configuration import config
from hgan.dm_datasets import all_systems, constant_physics, variable_physics


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
                [
                    resize(img, (config.arch.img_size, config.arch.img_size, nc))
                    for img in vid
                ]
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
        ]

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

        if img_size != config.arch.img_size:
            vid = np.asarray(
                [
                    resize(img, (config.arch.img_size, config.arch.img_size, nc))
                    for img in vid
                ]
            )

        # transpose each video to (nc, n_frames, img_size, img_size), and divide by 255
        vid = vid.transpose(3, 0, 1, 2)

        if config.video.normalize:
            vid = (vid - 0.5) / 0.5

        return vid.astype(np.float32)


class RealtimeDataset(Dataset):
    def __init__(self, system_name=None, num_frames=16, delta=1, train=True):
        jax.config.update("jax_enable_x64", True)

        if system_name is None:
            system_names = all_systems
            if config.experiment.system_color_constant:
                system_names = [x for x in system_names if "_COLORS" not in x]
            if config.experiment.system_friction:
                system_names = [x for x in system_names if "_FRICTION" in x]
        else:
            system_name = system_name.upper()
            if not config.experiment.system_color_constant:
                system_name += "_COLORS"
            if config.experiment.system_friction:
                system_name += "_FRICTION"

            assert system_name in all_systems, f"Unknown system {system_name}"
            system_names = [system_name]

        assert system_names, "No system selected"
        self.system_names = system_names

        self.num_frames = num_frames
        self.delta = delta
        self.train = train

        # Calling code in `get_fake_data` decides to randomly sample these `video_lengths`
        # to decide on the length of a fake video sample, so it is okay to initialize `video_lengths`
        # with a single value.
        self.video_lengths = [config.video.total_frames]

        self.generate_fn = {}  # Generate functions, keyed by system
        self.features = {}  # Fixed features across all trajectories, keyed by system

        for system_name in self.system_names:
            cls, config_ = getattr(datasets, system_name)
            config_dict = config_()

            # Tweak the physics parameters to our liking
            # TODO: Is this okay to do for speedup? Will this modify the characteristics of the experiment drastically?
            config_dict["image_resolution"] = config.arch.img_size
            _physics_key = (
                system_name.replace("COLORS", "").replace("FRICTION", "").rstrip("_")
            )
            if config.experiment.system_physics_constant:
                config_dict |= constant_physics[_physics_key]
            else:
                config_dict |= variable_physics[_physics_key]

            obj = cls(**config_dict)

            f = functools.partial(
                datasets.generate_sample,
                system=obj,
                dt=0.05,  # Blanchette 2021
                # num_steps is always 1 less than the no. of samples we wish to generate
                num_steps=config.video.total_frames - 1,
                steps_per_dt=1,
            )

            self.generate_fn[system_name] = f
            self.features[system_name] = f(0)["other"]

    def __len__(self):
        return 50_000 if self.train else 10_000  # Blanchette 2021

    def _physics_vector_from_data(self, data):
        props = np.zeros((config.arch.dp,))

        i = 0
        # note: dicts are ordered in py >= 3.7 so we have a deterministic order
        for k, v in data["other"].items():
            v = v.squeeze()
            if v.size == 1:
                props[i] = v.item()
                i += 1
                if i >= len(props):
                    return props
            else:
                for _v in v:
                    props[i] = _v
                    i += 1
                    if i >= len(props):
                        return props

        return props

    def __getitem__(self, item):
        system_name_index = np.random.choice(len(self.system_names))
        system_name = self.system_names[system_name_index]

        data = self.generate_fn[system_name](item)  # num_steps + 1, L, L, num_channels
        image = data["image"]
        # assert isinstance(image, jax.numpy.ndarray)
        # assert image.dtype == np.uint8

        vid = np.array(image / 255)

        vid = vid[:: self.delta]  # orig dt is 0.05
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - self.num_frames + 1)
        end = start + self.num_frames
        vid = vid[start:end]

        if img_size != config.arch.img_size:
            vid = np.asarray(
                [
                    resize(img, (config.arch.img_size, config.arch.img_size, nc))
                    for img in vid
                ]
            )

        # transpose each video to (nc, n_frames, img_size, img_size), and divide by 255
        vid = vid.transpose(3, 0, 1, 2)

        if config.video.normalize:
            vid = (vid - 0.5) / 0.5

        props = self._physics_vector_from_data(data)
        return vid.astype(np.float32), system_name_index, props

    def get_fake_labels(self, batch_size):
        fake_labels = Variable(
            torch.LongTensor(np.random.randint(0, len(self.system_names), batch_size))
        )
        return fake_labels


def build_dataloader(args):
    videos_dataset = get_dataset(args)
    if len(videos_dataset) == 0:
        raise RuntimeError("No videos found!")
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
    if config.experiment.hamiltonian_physics_rt:
        return RealtimeDataset(
            system_name=args.video_type, num_frames=config.video.frames
        )
    else:
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
    real_videos = real_system = real_props = None
    next_item = next(iter(videos_dataloader))
    if isinstance(next_item, (tuple, list)):
        real_videos = next_item[0]
        if len(next_item) > 1:
            real_system = next_item[1]
        if len(next_item) > 2:
            real_props = next_item[2]
    else:
        real_videos = next_item

    real_videos = real_videos.to(args.device)
    real_videos = Variable(real_videos)

    real_videos_frames = real_videos.shape[2]

    real_img = real_videos[:, :, np.random.randint(0, real_videos_frames), :, :]

    real_data = {
        "videos": real_videos,
        "img": real_img,
        "system": real_system,
        "props": real_props,
    }

    return real_data


def get_fake_data(args, dataset, video_lengths, rnn, gen_i, T=None):
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
    Z, dz, labels, props = get_latent_sample(args, dataset, rnn, n_frames)
    # trim => (batch_size, T, nz, 1, 1)
    Z = trim_noise(Z, T=T)
    Z_reshape = Z.contiguous().view(args.batch_size * T, args.nz, 1, 1)

    # generate videos
    fake_videos = gen_i(Z_reshape)

    fake_videos = fake_videos.view(
        args.batch_size, T, args.nc, args.img_size, args.img_size
    )
    # transpose => (batch_size, nc, T, img_size, img_size)
    fake_videos = fake_videos.transpose(2, 1)
    # img sampling
    fake_img = fake_videos[:, :, np.random.randint(0, T), :, :]

    fake_data = {
        "videos": fake_videos,
        "img": fake_img,
        "latent": Z,
        "dlatent": dz,
        "system": labels,
        "props": props,
    }

    return fake_data


def get_latent_sample(args, dataset, rnn, n_frames):
    if args.rnn_type in ["gru", "hnn_simple"]:
        return get_simple_sample(args, rnn, n_frames)
    elif args.rnn_type in ["hnn_phase_space"]:
        return get_phase_space_sample(args, dataset, rnn, n_frames)
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


def compute_phase_space_motion_vector(args, dataset, n_frames, rnn):
    eps = Variable(torch.randn(args.batch_size, args.d_E + args.d_L + args.d_P))
    eps = eps.to(args.device)

    rnn.initHidden(args.batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M, dz_M = rnn(eps, n_frames)
    z_M = z_M.transpose(1, 0)
    dz_M = dz_M.transpose(1, 0)
    if config.experiment.hamiltonian_physics_rt:
        fake_labels = dataset.get_fake_labels(args.batch_size)
    else:
        fake_labels = Variable(torch.LongTensor(np.zeros((args.batch_size,))))
    fake_props = eps[:, -(args.d_L + args.d_P) :] if (args.d_L + args.d_P) > 0 else None
    return z_M, dz_M, fake_labels, fake_props


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

    return z.view(args.batch_size, n_frames, args.nz, 1, 1), None, None, None


def get_phase_space_sample(args, dataset, rnn, n_frames):
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
    z_M, dz_M, labels, props = compute_phase_space_motion_vector(
        args, dataset, n_frames, rnn
    )
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)

    return z.view(args.batch_size, n_frames, args.nz, 1, 1), dz_M, labels, props


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
    return z.view(args.batch_size, n_frames, args.nz, 1, 1), None, None, None
