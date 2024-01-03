import os
import glob
import skvideo.io
from skimage.transform import resize
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
import jax
import functools
from hgan.dm_hamiltonian_dynamics_suite import datasets
from hgan.configuration import config
from hgan.dm_datasets import all_systems, constant_physics, variable_physics
from hgan.hgn.environments.environment_factory import EnvFactory


class AviDataset(Dataset):
    def __init__(self, datapath, T):
        self.T = T
        self.datapath = os.path.join(datapath, "resized_data")
        self.files = glob.glob(os.path.join(self.datapath, "*"))

        self.videos = self.get_videos()
        self.n_videos = len(self.videos)

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

        self.delta = delta
        self.datapath = os.path.join(datapath, train_test)
        self.files = glob.glob(os.path.join(self.datapath, "*.npy"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = os.path.join(self.datapath, f"{idx:06}.npy")

        vid = np.load(filename)
        vid = vid[:: self.delta]  # orig dt is 0.05
        n_frames, img_size, _, nc = vid.shape

        start = np.random.randint(0, n_frames - (self.T + 1))
        end = start + self.T
        vid = vid[start:end]

        vid = (
            np.asarray(
                [
                    resize(
                        img,
                        (config.experiment.img_size, config.experiment.img_size, nc),
                    )
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
    def __init__(self, *, datapath, num_frames, delta=1, train=True):
        train_test = "train" if train else "test"
        self.num_frames = num_frames
        self.delta = delta
        self.datapath = os.path.join(datapath, train_test)
        self.files = glob.glob(os.path.join(self.datapath, "*.npz"))

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

        if img_size != config.experiment.img_size:
            vid = np.asarray(
                [
                    resize(
                        img,
                        (config.experiment.img_size, config.experiment.img_size, nc),
                    )
                    for img in vid
                ]
            )

        # transpose each video to (nc, n_frames, img_size, img_size), and divide by 255
        vid = vid.transpose(3, 0, 1, 2)

        if config.video.normalize:
            vid = (vid - 0.5) / 0.5

        return vid.astype(np.float32)


class RealtimeDataset(Dataset):
    def __init__(
        self,
        *,
        ndim_physics=10,
        system_name=None,
        num_frames=16,
        delta=0.05,
        train=True,
        system_physics_constant=True,
        system_color_constant=True,
        system_friction=False,
        total_frames=100,
        img_size=32,
        normalize=False,
    ):

        jax.config.update("jax_enable_x64", True)

        if system_name is None:
            system_names = all_systems
            if system_color_constant:
                system_names = [x for x in system_names if "_COLORS" not in x]
            if system_friction:
                system_names = [x for x in system_names if "_FRICTION" in x]
        else:
            system_name = system_name.upper()
            if not system_color_constant:
                system_name += "_COLORS"
            if system_friction:
                system_name += "_FRICTION"

            assert system_name in all_systems, f"Unknown system {system_name}"
            system_names = [system_name]

        assert system_names, "No system selected"
        self.system_names = system_names

        self.num_frames = num_frames
        self.total_frames = total_frames
        self.delta = delta
        self.train = train
        self.ndim_physics = ndim_physics
        self.img_size = img_size
        self.normalize = normalize

        self.generate_fn = {}  # Generate functions, keyed by system
        self.features = {}  # Fixed features across all trajectories, keyed by system

        for system_name in self.system_names:
            cls, config_ = getattr(datasets, system_name)
            config_dict = config_()

            # Tweak the physics parameters to our liking
            # TODO: Is this okay to do for speedup? Will this modify the characteristics of the experiment drastically?
            config_dict["image_resolution"] = img_size
            _physics_key = (
                system_name.replace("COLORS", "").replace("FRICTION", "").rstrip("_")
            )
            if system_physics_constant:
                config_dict |= constant_physics[_physics_key]
            else:
                config_dict |= variable_physics[_physics_key]

            obj = cls(**config_dict)

            f = functools.partial(
                datasets.generate_sample,
                system=obj,
                dt=0.05,  # Blanchette 2021
                # num_steps is always 1 less than the no. of samples we wish to generate
                num_steps=total_frames - 1,
                steps_per_dt=1,
            )

            self.generate_fn[system_name] = f
            self.features[system_name] = f(0)["other"]

    def __len__(self):
        return 50_000 if self.train else 10_000  # Blanchette 2021

    def _physics_vector_from_data(self, data):
        ndim_physics = self.ndim_physics
        if ndim_physics <= 0:
            return 0

        props = np.zeros((ndim_physics,))

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

        if img_size != self.img_size:
            vid = np.asarray(
                [
                    resize(
                        img,
                        (
                            self.img_size,
                            self.img_size,
                            nc,
                        ),
                    )
                    for img in vid
                ]
            )

        # transpose each video to (nc, n_frames, img_size, img_size)
        vid = vid.transpose(3, 0, 1, 2)

        if self.normalize:
            vid = (vid - 0.5) / 0.5

        props = self._physics_vector_from_data(data)
        return vid.astype(np.float32), system_name_index, props

    def get_fake_labels(self, batch_size):
        fake_labels = Variable(
            torch.LongTensor(np.random.randint(0, len(self.system_names), batch_size))
        )
        return fake_labels


class HGNRealtimeDataset(Dataset):
    def __init__(
        self,
        *,
        ndim_physics=10,
        system_name=None,
        num_frames=16,
        delta=0.05,
        train=True,
        system_physics_constant=True,
        system_color_constant=True,
        system_friction=False,
        total_frames=100,
        img_size=32,
        normalize=False,
    ):

        system_name_to_hgn_name_and_args = {
            "mass_spring": ("Spring", {"mass": 0.5, "elastic_cst": 2.0}),
            "pendulum": ("Pendulum", {"mass": 0.5, "length": 1.0, "g": 3.0}),
            "double_pendulum": (
                "ChaoticPendulum",
                {"mass": 1.0, "length": 1.0, "g": 3.0},
            ),
            "two_body": (
                "NObjectGravity",
                {"mass": [1.0, 1.0], "g": 1.0, "orbit_noise": 0.1},
            ),
            "three_body": (
                "NObjectGravity",
                {"mass": [1.0, 1.0, 1.0], "g": 1.0, "orbit_noise": 0.1},
            ),
        }
        self.system_name, self.system_args = system_name_to_hgn_name_and_args[
            system_name
        ]
        self.system_names = [self.system_name]  # TODO

        self.num_frames = num_frames
        self.total_frames = total_frames
        self.delta = delta
        self.train = train
        self.ndim_physics = ndim_physics
        self.img_size = img_size
        self.normalize = normalize

    def __len__(self):
        return 50_000 if self.train else 10_000  # Blanchette 2021

    def __getitem__(self, item):
        system = EnvFactory.get_environment(self.system_name, **self.system_args)
        # We're not using self.total_frames here at all, since we only want self.num_frames from
        # the rollout, and the rollouts are randomly initialized anyway.
        vids = system.sample_random_rollouts(
            number_of_frames=self.num_frames,
            delta_time=self.delta,
            number_of_rollouts=1,
            img_size=self.img_size,
            noise_level=0.1,
            radius_bound="auto",
            color=True,
            seed=None,
        )

        vid = vids[0]

        # transpose each video to (nc, n_frames, img_size, img_size)
        vid = vid.transpose(3, 0, 1, 2)

        if self.normalize:
            vid = (vid - 0.5) / 0.5

        props = system.physical_properties(vec_length=self.ndim_physics)
        system_name_index = 0
        return vid.astype(np.float32), system_name_index, props

    def get_fake_labels(self, batch_size):
        fake_labels = Variable(
            torch.LongTensor(np.random.randint(0, len(self.system_names), batch_size))
        )
        return fake_labels
