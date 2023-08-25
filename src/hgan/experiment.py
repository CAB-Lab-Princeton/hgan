import os.path
import torch
from torch.utils.data import DataLoader
from hgan.configuration import save_config
from hgan.models import GRU, HNNSimple, HNNPhaseSpace, HNNMass
from hgan.train import setup_reproducibility, build_models, train
from hgan.dataset import RealtimeDataset


class Experiment:
    def __init__(self, config):
        self.config = config
        self._init_derived_attributes(self.config)

    def _init_derived_attributes(self, config):
        if config.experiment.system_name is not None:
            self.datapath = os.path.join(
                config.paths.input, config.experiment.system_name
            )
        else:
            self.datapath = None

        self.rnn = {
            "gru": GRU,
            "hnn_simple": HNNSimple,
            "hnn_phase_space": HNNPhaseSpace,
            "hnn_mass": HNNMass,
        }[config.experiment.architecture]

        if config.experiment.gpu is None or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = f"cuda:{config.experiment.gpu}"

        self.criterion = torch.nn.BCELoss().to(self.device)
        self.label = torch.FloatTensor().to(self.device)

        self.img_size = config.arch.img_size
        self.nc = config.arch.nc
        self.ndf = config.arch.ndf
        self.ngf = config.arch.ngf

        self.hidden_size = config.arch.hidden_size

        self.d_E = config.arch.de
        self.d_C = config.arch.dc
        # Dimension of motion vector (since (p, q) vectors are concatenated, same as d_E)
        self.d_M = self.d_E
        self.d_L = config.arch.dl
        self.d_P = config.arch.dp

        self.lr = config.experiment.learning_rate
        self.betas = tuple(float(b) for b in config.experiment.betas.split(","))

        self.q_size = int(self.d_M / 2)
        self.d_N = int(self.q_size**2)

        self.nz = (
            self.q_size + int((self.d_N + self.q_size) / 2) + self.d_C
            if config.experiment.architecture == "hnn_mass"
            else self.d_C + self.d_M
        )

    def get_dataloader(self, batch_size):
        if self.config.experiment.hamiltonian_physics_rt:
            dataset = RealtimeDataset(
                config=self.config,
                system_name=self.config.experiment.system_name,
                num_frames=self.config.video.frames,
            )
        else:
            # dataset = ToyPhysicsDatasetNPZ(self.datapath, num_frames=self.config.video.frames)
            raise NotImplementedError

        if len(dataset) == 0:
            raise RuntimeError("No videos found!")

        return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    def run(self):

        save_config(self.config.paths.output)

        setup_reproducibility(seed=self.config.experiment.seed)
        dataloader = self.get_dataloader(batch_size=self.config.experiment.batch_size)

        models = build_models(
            rnn=self.rnn,
            rnn_type=self.config.experiment.architecture,
            nc=self.nc,
            ndf=self.ndf,
            ngpu=self.config.experiment.gpu,
            ngf=self.ngf,
            nz=self.nz,
            d_E=self.d_E,
            d_L=self.d_L,
            d_P=self.d_P,
            hidden_size=self.hidden_size,
            device=self.device,
        )

        train(
            rnn_type=self.config.experiment.architecture,
            label=self.label,
            criterion=self.criterion,
            seed=self.config.experiment.seed,
            lr=self.lr,
            betas=self.betas,
            batch_size=self.config.experiment.batch_size,
            q_size=self.q_size,
            cyclic_coord_loss=self.config.experiment.cyclic_coord_loss,
            d_C=self.d_C,
            d_E=self.d_E,
            d_L=self.d_L,
            d_P=self.d_P,
            d_N=self.d_N,
            nz=self.nz,
            nc=self.nc,
            img_size=self.img_size,
            trained_models_dir=self.config.paths.output,
            generated_videos_dir=self.config.paths.output,
            device=self.device,
            niter=self.config.experiment.n_epoch,
            print_every=self.config.experiment.print_every,
            save_output_every=self.config.experiment.save_video_every,
            save_model_every=self.config.experiment.save_model_every,
            models=models,
            videos_dataloader=dataloader,
            retrain=self.config.experiment.retrain,
        )
