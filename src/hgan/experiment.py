import logging
import os.path
import time
import glob
import numpy as np
import skvideo.io
from skimage.transform import resize
import importlib
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import hgan.data
from hgan.configuration import save_config
from hgan.models import GRU, HNNSimple, HNNPhaseSpace, HNNMass
from hgan.dataset import RealtimeDataset, HGNRealtimeDataset, ToyPhysicsDatasetNPZ
from hgan.utils import setup_reproducibility, timeSince
from hgan.fvd import compute_fvd
from hgan.models import Discriminator_I, Discriminator_V, Generator_I
from hgan.updates import update_models


logger = logging.getLogger(__name__)


class Experiment:
    def __init__(self, config):
        self.dataloader = None
        self.model_names = (
            "Di",
            "Dv",
            "Gi",
            "rnn",
            "optim_Di",
            "optim_Dv",
            "optim_Gi",
            "optim_rnn",
        )
        self.Di = self.Dv = self.Gi = self.rnn = None
        self.optim_Di = self.optim_Dv = self.optim_Gi = self.optim_rnn = None
        self.config = config

        self._init_derived_attributes(self.config)
        self._init_dataloader(self.config)
        self._init_models(self.config)

    def _init_derived_attributes(self, config):

        # Make all keys available in config.experiment as attributes in this object, for convenience
        for k, v in config.experiment.items():
            setattr(self, k, v)

        # Only 1 GPU supported for now
        self.ngpu = 1

        if config.experiment.system_name is not None and config.paths.input is not None:
            self.datapath = os.path.join(
                config.paths.input, config.experiment.system_name
            )
        else:
            self.datapath = None

        if config.experiment.gpu is None or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = f"cuda:{config.experiment.gpu}"

        self.criterion = torch.nn.BCELoss().to(self.device)
        self.label = torch.FloatTensor().to(self.device)

        self.ndim_p = self.ndim_q = int(self.ndim_epsilon / 2)
        self.betas = tuple(float(b) for b in config.experiment.betas.split(","))

        self.ndim_q2 = int(self.ndim_q**2)

        if config.experiment.architecture == "hnn_mass":
            self.nz = (
                self.ndim_q + int((self.ndim_q2 + self.ndim_q) / 2) + self.ndim_content
            )
        else:
            self.nz = self.ndim_content + self.ndim_epsilon

    def _init_dataloader(self, config):
        if config.experiment.rt_data_generator == "hgn":
            dataset = HGNRealtimeDataset(
                ndim_physics=config.experiment.ndim_physics,
                system_name=config.experiment.system_name,
                num_frames=config.video.generator_frames,
                delta=1,
                train=True,
                system_physics_constant=config.experiment.system_physics_constant,
                system_color_constant=config.experiment.system_color_constant,
                system_friction=config.experiment.system_friction,
                total_frames=config.video.real_total_frames,
                img_size=config.experiment.img_size,
                normalize=config.video.normalize,
            )
        elif config.experiment.rt_data_generator == "dm":
            dataset = RealtimeDataset(
                system_name=config.experiment.system_name,
                num_frames=config.video.frames,
            )
        else:
            dataset = ToyPhysicsDatasetNPZ(
                datapath=self.datapath, num_frames=config.video.generator_frames
            )

        if len(dataset) == 0:
            raise RuntimeError("No videos found!")

        self.dataloader = DataLoader(
            dataset,
            batch_size=config.experiment.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def _init_models(self, config):
        self.Di = Discriminator_I(
            self.ndim_channel, self.ndim_discriminator_filter, ngpu=self.ngpu
        ).to(self.device)
        self.Dv = Discriminator_V(
            self.ndim_channel,
            self.ndim_discriminator_filter,
            T=config.video.discriminator_frames,
        ).to(self.device)
        self.Gi = Generator_I(
            self.ndim_channel, self.ndim_generator_filter, self.nz, ngpu=self.ngpu
        ).to(self.device)

        rnn_class = {
            "gru": GRU,
            "hnn_simple": HNNSimple,
            "hnn_phase_space": HNNPhaseSpace,
            "hnn_mass": HNNMass,
        }[config.experiment.architecture]

        if config.experiment.architecture in ("hnn_phase_space", "hnn_mass"):
            self.rnn = rnn_class(
                device=self.device,
                input_size=self.ndim_epsilon + self.ndim_label + self.ndim_physics,
                hidden_size=self.hidden_size,
                output_size=self.ndim_epsilon,
                ndim_physics=self.ndim_physics,
                ndim_label=self.ndim_label,
            ).to(self.device)
        else:
            self.rnn = rnn_class(
                device=self.device,
                input_size=self.ndim_epsilon,
                hidden_size=self.hidden_size,
            ).to(self.device)

        self.rnn.initWeight()

        self.optim_Di = torch.optim.Adam(
            self.Di.parameters(), lr=self.learning_rate, betas=self.betas
        )
        self.optim_Dv = torch.optim.Adam(
            self.Dv.parameters(), lr=self.learning_rate, betas=self.betas
        )
        self.optim_Gi = torch.optim.Adam(
            self.Gi.parameters(), lr=self.learning_rate, betas=self.betas
        )
        self.optim_rnn = torch.optim.Adam(
            self.rnn.parameters(), lr=self.learning_rate, betas=self.betas
        )

    def saved_epochs(self):
        saved_pths = sorted(glob.glob(self.config.paths.output + "/Di_*.pth"))
        filenames = [os.path.splitext(os.path.basename(p))[0] for p in saved_pths]
        epochs = [int(filename.split("_")[-1]) for filename in filenames]
        return epochs

    def load_epoch(self, epoch=None):
        if epoch is None:
            saved_epochs = self.saved_epochs()
            if not saved_epochs:
                return 0
            epoch = saved_epochs[-1]

        for which in self.model_names:
            file_path = os.path.join(
                self.config.paths.output, f"{which}_{epoch:0>6}.pth"
            )
            model = getattr(self, which)
            model.load_state_dict(torch.load(file_path))

        return epoch

    def eval(self):
        for which in self.model_names:
            if not which.startswith("optim"):
                model = getattr(self, which)
                model.eval()

    def save_video(self, folder, video, epoch, prefix="video_"):
        os.makedirs(folder, exist_ok=True)
        outputdata = video * 255
        outputdata = outputdata.astype(np.uint8)
        file_path = os.path.join(folder, f"{prefix}{epoch:0>6}.mp4")
        skvideo.io.vwrite(file_path, outputdata, verbosity=0)

    def save_epoch(self, epoch):
        for which in self.model_names:
            file_path = os.path.join(
                self.config.paths.output, f"{which}_{epoch:0>6}.pth"
            )
            model = getattr(self, which)
            torch.save(model.state_dict(), file_path)

    def get_random_content_vector(self, batch_size, d_C, device, n_frames):
        z_C = Variable(torch.randn(batch_size, d_C))
        #  repeat z_C to (batch_size, n_frames, d_C)
        z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
        z_C = z_C.to(device)

        return z_C

    def compute_phase_space_motion_vector(
        self, batch_size, d_E, d_L, d_P, device, dataset, n_frames, rnn
    ):
        eps = Variable(torch.randn(batch_size, d_E + d_L + d_P))
        eps = eps.to(device)

        rnn.initHidden(batch_size)
        # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
        z_M, dz_M = rnn(eps, n_frames)
        z_M = z_M.transpose(1, 0)
        if dz_M is not None:
            dz_M = dz_M.transpose(1, 0)
        if self.rt_data_generator is not None:
            fake_labels = dataset.get_fake_labels(batch_size)
        else:
            fake_labels = Variable(torch.LongTensor(np.zeros((batch_size,))))
        fake_props = eps[:, -(d_L + d_P) :] if (d_L + d_P) > 0 else None
        return z_M, dz_M, fake_labels, fake_props

    def get_phase_space_sample(
        self, batch_size, d_C, d_E, d_L, d_P, device, nz, dataset, rnn, n_frames
    ):
        z_C = self.get_random_content_vector(batch_size, d_C, device, n_frames)
        z_M, dz_M, labels, props = self.compute_phase_space_motion_vector(
            batch_size=batch_size,
            d_E=d_E,
            d_L=d_L,
            d_P=d_P,
            device=device,
            dataset=dataset,
            n_frames=n_frames,
            rnn=rnn,
        )
        z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)

        return z.view(batch_size, n_frames, nz, 1, 1), dz_M, labels, props

    def compute_simple_motion_vector(self, batch_size, d_E, device, n_frames, rnn):
        eps = Variable(torch.randn(batch_size, d_E))
        eps = eps.to(device)

        rnn.initHidden(batch_size)
        # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
        z_M = rnn(eps, n_frames).transpose(1, 0)
        return z_M

    def compute_mass_motion_vector(self, batch_size, d_E, d_N, device, n_frames, rnn):
        eps = Variable(torch.randn(batch_size, d_E))
        Z_mass = Variable(torch.randn(batch_size, d_N))

        eps = eps.to(device)
        Z_mass = Z_mass.to(device)

        rnn.initHidden(batch_size)
        # notice that 1st dim of hnn outputs is seq_len, 2nd is batch_size
        z_M, z_mass = rnn(eps, Z_mass, n_frames)

        z_M = z_M.transpose(1, 0)
        z_mass = z_mass.unsqueeze(1).repeat(1, n_frames, 1)

        return z_M, z_mass

    def get_simple_sample(self, batch_size, d_C, d_E, nz, device, rnn, n_frames):
        z_C = self.get_random_content_vector(batch_size, d_C, device, n_frames)
        z_M = self.compute_simple_motion_vector(batch_size, d_E, device, n_frames, rnn)
        z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)

        return z.view(batch_size, n_frames, nz, 1, 1), None, None, None

    def get_mass_sample(self, batch_size, d_C, d_E, d_N, device, nz, rnn, n_frames):
        z_C = self.get_random_content_vector(batch_size, d_C, device, n_frames)
        z_M, z_mass = self.compute_mass_motion_vector(
            batch_size, d_E, d_N, device, n_frames, rnn
        )
        z = torch.cat((z_M, z_mass, z_C), 2)  # z.size() => (batch_size, n_frames, nz)
        return z.view(batch_size, n_frames, nz, 1, 1), None, None, None

    def get_latent_sample(
        self,
        batch_size,
        n_frames,
    ):
        # TODO: Do this through inheritance
        if self.architecture in ("gru", "hnn_simple"):
            return self.get_simple_sample(
                batch_size,
                self.ndim_content,
                self.ndim_epsilon,
                self.nz,
                self.device,
                self.rnn,
                n_frames,
            )
        elif self.architecture in ("hnn_phase_space",):
            return self.get_phase_space_sample(
                batch_size=batch_size,
                d_C=self.ndim_content,
                d_E=self.ndim_epsilon,
                d_L=self.ndim_label,
                d_P=self.ndim_physics,
                device=self.device,
                nz=self.nz,
                dataset=self.dataloader.dataset,
                rnn=self.rnn,
                n_frames=n_frames,
            )
        else:
            return self.get_mass_sample(
                batch_size=batch_size,
                d_C=self.ndim_content,
                d_E=self.ndim_epsilon,
                d_N=self.ndim_q2,
                device=self.device,
                nz=self.nz,
                rnn=self.rnn,
                n_frames=n_frames,
            )

    def trim_video(self, video, n_frame):
        # Trim a (batch_size, T, ...) video to (batch_size, n_frame, ...)
        start = np.random.randint(0, video.size(1) - n_frame + 1)
        end = start + n_frame
        return video[:, start:end, ...]

    def get_fake_data(self, n_frames=None):
        n_frames = n_frames or self.config.video.generator_frames
        # Z.size() => (batch_size, n_frames, nz, 1, 1)
        Z, dz, labels, props = self.get_latent_sample(
            batch_size=self.batch_size,
            n_frames=n_frames,
        )
        # trim => (batch_size, T, nz, 1, 1)
        Z = self.trim_video(video=Z, n_frame=n_frames)
        Z_reshape = Z.contiguous().view(self.batch_size * n_frames, self.nz, 1, 1)

        fake_videos = self.Gi(Z_reshape)

        fake_videos = fake_videos.view(
            self.batch_size, n_frames, self.ndim_channel, self.img_size, self.img_size
        )
        # transpose => (batch_size, nc, T, img_size, img_size)
        fake_videos = fake_videos.transpose(2, 1)
        # img sampling
        fake_img = fake_videos[:, :, np.random.randint(0, n_frames), :, :]

        fake_data = {
            "videos": fake_videos,
            "img": fake_img,
            "latent": Z,
            "dlatent": dz,
            "system": labels,
            "props": props,
        }

        return fake_data

    def save_fake_images(self, generated_img_path, n=1, video_length=50):
        for i in range(n):
            fake_data = self.get_fake_data()
            # (batch_size, T, nc, img_size, img_size)
            fake_data_np = (
                fake_data["videos"].permute(0, 2, 1, 3, 4).detach().cpu().numpy()
            )
            filename = generated_img_path + str(i).zfill(4)
            np.save(filename, fake_data_np)

    def get_real_data(self, device=None, dataloader=None):
        device = device or self.device
        dataloader = dataloader or self.dataloader
        real_system = real_props = None
        next_item = next(iter(dataloader))
        if isinstance(next_item, (tuple, list)):
            real_videos = next_item[0]
            if len(next_item) > 1:
                real_system = next_item[1]
            if len(next_item) > 2:
                real_props = next_item[2]
        else:
            real_videos = next_item

        real_videos = real_videos.to(
            device
        )  # (batch_size, ndim_channels, n_frames, img_size, img_size)
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

    def fvd(self, real_videos=None, fake_videos=None, max_videos=None, device="cpu"):

        if real_videos is None:
            real_videos = self.get_real_data()[
                "videos"
            ]  # (batch_size, n_channels, n_frames, height, width)
        if fake_videos is None:
            fake_videos = self.get_fake_data()[
                "videos"
            ]  # (batch_size, n_channels, n_frames, height, width)

        # Use shape (batch_size, n_frames, n_channels, height, width)
        real_videos = real_videos.permute(0, 2, 1, 3, 4).detach().cpu().numpy()
        fake_videos = fake_videos.permute(0, 2, 1, 3, 4).detach().cpu().numpy()

        with importlib.resources.path(hgan.data, "i3d_torchscript.pt") as i3d_path:
            detector = torch.jit.load(i3d_path).eval().to(device)

        batch_size, num_frames, num_channels, height, width = real_videos.shape
        assert num_channels == 3, "Inputs should be 3 channels"

        resized_real_videos = []
        for vid in real_videos[:max_videos]:
            resized_video = np.asarray([resize(img, (3, 224, 224)) for img in vid])
            resized_real_videos.append(resized_video)
        resized_real_videos = np.array(resized_real_videos)

        resized_fake_videos = []
        for vid in fake_videos[:max_videos]:
            resized_video = np.asarray([resize(img, (3, 224, 224)) for img in vid])
            resized_fake_videos.append(resized_video)
        resized_fake_videos = np.array(resized_fake_videos)

        # detector expects inputs of shape (batch_size, num_channels, num_frames, height, width)
        resized_real_videos = (
            torch.from_numpy(resized_real_videos).to(device).permute(0, 2, 1, 3, 4)
        )
        resized_fake_videos = (
            torch.from_numpy(resized_fake_videos).to(device).permute(0, 2, 1, 3, 4)
        )

        detector_kwargs = {
            "rescale": False,
            "resize": False,
            "return_features": True,  # Return raw features before the softmax layer.
        }
        feats_real = (
            detector(resized_real_videos, **detector_kwargs).detach().cpu().numpy()
        )
        feats_fake = (
            detector(resized_fake_videos, **detector_kwargs).detach().cpu().numpy()
        )

        fvd = compute_fvd(real_activations=feats_real, generated_activations=feats_fake)
        return fvd

    def train_step(self):
        real_data = self.get_real_data()
        fake_data = self.get_fake_data()

        err, mean = update_models(
            rnn_type=self.architecture,
            label=self.label,
            criterion=self.criterion,
            q_size=self.ndim_q,
            batch_size=self.batch_size,
            cyclic_coord_loss=self.cyclic_coord_loss,
            r1_gamma=self.r1_gamma,
            model_di=self.Di,
            model_dv=self.Dv,
            model_gi=self.Gi,
            model_rnn=self.rnn,
            optim_di=self.optim_Di,
            optim_dv=self.optim_Dv,
            optim_gi=self.optim_Gi,
            optim_rnn=self.optim_rnn,
            real_data=real_data,
            fake_data=fake_data,
        )

        return err, mean, real_data, fake_data

    def train(self):

        for which in self.model_names:
            if not which.startswith("optim"):
                model = getattr(self, which)
                model.eval()

        save_config(self.config.paths.output)
        setup_reproducibility(seed=self.seed)

        if self.retrain:
            start_epoch = 0
        else:
            start_epoch = self.load_epoch()

        start_time = time.time()
        for epoch in range(start_epoch + 1, self.n_epoch + 1):
            err, mean, real_data, fake_data = self.train_step()

            real_videos = real_data["videos"]
            first_real_video = real_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0)
            fake_videos = fake_data["videos"]
            first_fake_video = fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0)

            last_epoch = epoch == self.n_epoch

            if epoch % self.calculate_fvd_every == 0 or last_epoch:
                fvd = self.fvd(real_videos=real_videos, fake_videos=fake_videos)
                logger.info(f"FVD = {fvd}")

            if epoch % self.print_every == 0 or last_epoch:
                logger.info(
                    "[%d/%d] (%s) Loss_Di: %.4f Loss_Dv: %.4f Loss_Gi: %.4f Loss_Gv: %.4f Di_real_mean %.4f Di_fake_mean %.4f Dv_real_mean %.4f Dv_fake_mean %.4f"
                    % (
                        epoch,
                        self.n_epoch,
                        timeSince(start_time),
                        err["Di"],
                        err["Dv"],
                        err["Gi"],
                        err["Gv"],
                        mean["Di_real"],
                        mean["Di_fake"],
                        mean["Dv_real"],
                        mean["Dv_fake"],
                    )
                )

            if epoch % self.save_real_video_every == 0 or last_epoch:
                self.save_video(
                    self.config.paths.output, first_fake_video, epoch, prefix="fake_"
                )

            if epoch % self.save_fake_video_every == 0 or last_epoch:
                self.save_video(
                    self.config.paths.output, first_real_video, epoch, prefix="real_"
                )

            if epoch % self.save_model_every == 0 or last_epoch:
                self.save_epoch(epoch)
