import os.path
import time
import glob
import numpy as np
import skvideo.io
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from hgan.configuration import save_config
from hgan.models import GRU, HNNSimple, HNNPhaseSpace, HNNMass
from hgan.dataset import RealtimeDataset, ToyPhysicsDatasetNPZ
from hgan.utils import setup_reproducibility, timeSince, trim_noise
from hgan.models import Discriminator_I, Discriminator_V, Generator_I
from hgan.updates import update_models


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
        if config.experiment.hamiltonian_physics_rt:
            dataset = RealtimeDataset(
                config=config,
                system_name=config.experiment.system_name,
                num_frames=config.video.frames,
            )
        else:
            dataset = ToyPhysicsDatasetNPZ(
                config=config, datapath=self.datapath, num_frames=config.video.frames
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
            T=config.video.frames,
            ngpu=self.ngpu,
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

    def save_video(self, folder, video, epoch):
        os.makedirs(folder, exist_ok=True)
        outputdata = video * 255
        outputdata = outputdata.astype(np.uint8)
        file_path = os.path.join(folder, f"video_{epoch:0>6}.mp4")
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
        dz_M = dz_M.transpose(1, 0)
        if self.hamiltonian_physics_rt:
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
        rnn_type,
        batch_size,
        d_C,
        d_E,
        d_L,
        d_P,
        d_N,
        device,
        nz,
        dataset,
        rnn,
        n_frames,
    ):
        # TODO: Do this through inheritance
        if rnn_type in ("gru", "hnn_simple"):
            return self.get_simple_sample(
                batch_size, d_C, d_E, nz, device, rnn, n_frames
            )
        elif rnn_type in ["hnn_phase_space"]:
            return self.get_phase_space_sample(
                batch_size=batch_size,
                d_C=d_C,
                d_E=d_E,
                d_L=d_L,
                d_P=d_P,
                device=device,
                nz=nz,
                dataset=dataset,
                rnn=rnn,
                n_frames=n_frames,
            )
        else:
            return self.get_mass_sample(
                batch_size=batch_size,
                d_C=d_C,
                d_E=d_E,
                d_N=d_N,
                device=device,
                nz=nz,
                rnn=rnn,
                n_frames=n_frames,
            )

    def get_fake_data(
        self,
        nz,
        nc,
        img_size,
        dataset,
        video_lengths,
    ):
        T = self.config.video.frames

        n_videos = len(video_lengths)
        idx = np.random.randint(0, n_videos)
        n_frames = video_lengths[idx]

        # Z.size() => (batch_size, n_frames, nz, 1, 1)
        Z, dz, labels, props = self.get_latent_sample(
            rnn_type=self.architecture,
            batch_size=self.batch_size,
            d_C=self.ndim_content,
            d_E=self.ndim_epsilon,
            d_L=self.ndim_label,
            d_P=self.ndim_physics,
            d_N=self.ndim_q2,
            device=self.device,
            nz=nz,
            dataset=dataset,
            rnn=self.rnn,
            n_frames=n_frames,
        )
        # trim => (batch_size, T, nz, 1, 1)
        Z = trim_noise(Z, T=T)
        Z_reshape = Z.contiguous().view(self.batch_size * T, nz, 1, 1)

        fake_videos = self.Gi(Z_reshape)

        fake_videos = fake_videos.view(self.batch_size, T, nc, img_size, img_size)
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

    def get_real_data(self, device, videos_dataloader):
        real_system = real_props = None
        next_item = next(iter(videos_dataloader))
        if isinstance(next_item, (tuple, list)):
            real_videos = next_item[0]
            if len(next_item) > 1:
                real_system = next_item[1]
            if len(next_item) > 2:
                real_props = next_item[2]
        else:
            real_videos = next_item

        real_videos = real_videos.to(device)
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

    def train_step(self):
        dataset = self.dataloader.dataset
        video_lengths = self.dataloader.dataset.video_lengths

        real_data = self.get_real_data(
            device=self.device, videos_dataloader=self.dataloader
        )
        fake_data = self.get_fake_data(
            nz=self.nz,
            nc=self.ndim_channel,
            img_size=self.img_size,
            dataset=dataset,
            video_lengths=video_lengths,
        )

        err, mean = update_models(
            rnn_type=self.architecture,
            label=self.label,
            criterion=self.criterion,
            q_size=self.ndim_q,
            batch_size=self.batch_size,
            cyclic_coord_loss=self.cyclic_coord_loss,
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

        return err, mean, fake_data["videos"]

    def train(self):
        save_config(self.config.paths.output)
        setup_reproducibility(seed=self.seed)

        if self.retrain:
            start_epoch = 0
        else:
            start_epoch = self.load_epoch()

        start_time = time.time()
        for epoch in range(start_epoch + 1, self.n_epoch + 1):
            err, mean, fake_videos = self.train_step()

            last_epoch = epoch == self.n_epoch

            if epoch % self.print_every == 0 or last_epoch:
                print(
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

            if epoch % self.save_video_every == 0 or last_epoch:
                video = fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0)
                self.save_video(self.config.paths.output, video, epoch)

            if epoch % self.save_model_every == 0 or last_epoch:
                self.save_epoch(epoch)
