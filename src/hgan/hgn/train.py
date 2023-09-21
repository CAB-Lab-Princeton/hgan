import os
import warnings
import importlib
import yaml
import numpy as np
import torch
import tqdm
import tempfile

import hgan.data
from hgan.hgn.utilities.training_logger import TrainingLogger
from hgan.hgn.utilities import loader
from hgan.hgn.utilities.loader import (
    load_hgn,
    get_online_dataloaders,
    get_offline_dataloaders,
)
from hgan.hgn.utilities.losses import reconstruction_loss, kld_loss, geco_constraint
from hgan.hgn.utilities.statistics import mean_confidence_interval


class HgnTrainer:
    def __init__(self, params):

        self.params = params

        # Set device
        self.device = params["device"]
        if "cuda" in self.device and not torch.cuda.is_available():
            warnings.warn(
                "Warning! Set to train in GPU but cuda is not available. Device is set to CPU."
            )
            self.device = "cpu"

        # Get dtype, will raise a 'module 'torch' has no attribute' if there is a typo
        self.dtype = torch.__getattribute__(params["networks"]["dtype"])

        # Load hgn from parameters to device
        self.hgn = load_hgn(params=self.params, device=self.device, dtype=self.dtype)
        if "load_path" in self.params:
            self.load_and_reset(self.params, self.device, self.dtype)

        # Either generate data on-the-fly or load the data from disk
        if "train_data" in self.params["dataset"]:
            print("Training with OFFLINE data...")
            self.train_data_loader, self.test_data_loader = get_offline_dataloaders(
                self.params
            )
        else:
            print("Training with ONLINE data...")
            self.train_data_loader, self.test_data_loader = get_online_dataloaders(
                self.params
            )

        # Initialize training logger
        self.training_logger = TrainingLogger(
            hyper_params=self.params,
            loss_freq=100,
            rollout_freq=1000,
            model_freq=1000,
            log_dir=params["log_dir"],
        )

        # Initialize tensorboard writer
        self.model_save_file = os.path.join(
            self.params["model_save_dir"], self.params["experiment_id"]
        )

        # Define optimization modules
        optim_params = [
            {
                "params": self.hgn.encoder.parameters(),
                "lr": params["optimization"]["encoder_lr"],
            },
            {
                "params": self.hgn.transformer.parameters(),
                "lr": params["optimization"]["transformer_lr"],
            },
            {
                "params": self.hgn.hnn.parameters(),
                "lr": params["optimization"]["hnn_lr"],
            },
            {
                "params": self.hgn.decoder.parameters(),
                "lr": params["optimization"]["decoder_lr"],
            },
        ]
        self.optimizer = torch.optim.Adam(optim_params)

    def load_and_reset(self, params, device, dtype):
        """Load the HGN from the path specified in params['load_path'] and reset the networks in
        params['reset'].

        Args:
            params (dict): Dictionary with all the necessary parameters to load the networks.
            device (str): 'gpu:N' or 'cpu'
            dtype (torch.dtype): Data type to be used in computations.
        """
        self.hgn.load(params["load_path"])
        if "reset" in params:
            if isinstance(params["reset"], list):
                for net in params["reset"]:
                    assert net in ["encoder", "decoder", "hamiltonian", "transformer"]
            else:
                assert params["reset"] in [
                    "encoder",
                    "decoder",
                    "hamiltonian",
                    "transformer",
                ]
            if "encoder" in params["reset"]:
                self.hgn.encoder = loader.instantiate_encoder(params, device, dtype)
            if "decoder" in params["reset"]:
                self.hgn.decoder = loader.instantiate_decoder(params, device, dtype)
            if "transformer" in params["reset"]:
                self.hgn.transformer = loader.instantiate_transformer(
                    params, device, dtype
                )
            if "hamiltonian" in params["reset"]:
                self.hgn.hnn = loader.instantiate_hamiltonian(params, device, dtype)

    def training_step(self, rollouts):
        """Perform a training step with the given rollouts batch.

        Args:
            rollouts (torch.Tensor): Tensor of shape (batch_size, seq_len, channels, height, width)
                corresponding to a batch of sampled rollouts.

        Returns:
            A dictionary of losses and the model's prediction of the rollout. The reconstruction loss and
            KL divergence are floats and prediction is the HGNResult object with data of the forward pass.
        """
        self.optimizer.zero_grad()

        rollout_len = rollouts.shape[1]
        input_frames = self.params["optimization"]["input_frames"]
        assert (
            input_frames <= rollout_len
        )  # optimization.use_steps must be smaller (or equal) to rollout.sequence_length
        roll = rollouts[:, :input_frames]

        hgn_output = self.hgn.forward(
            rollout_batch=roll, n_steps=rollout_len - input_frames
        )
        target = rollouts[
            :, input_frames - 1 :
        ]  # Fit first input_frames and try to predict the last + the next (rollout_len - input_frames)
        prediction = hgn_output.reconstructed_rollout

        if self.params["networks"]["variational"]:
            tol = self.params["geco"]["tol"]
            alpha = self.params["geco"]["alpha"]
            lagrange_mult_param = self.params["geco"]["lagrange_multiplier_param"]

            C, rec_loss = geco_constraint(target, prediction, tol)  # C has gradient

            # Compute moving average of constraint C (without gradient)
            if self.C_ma is None:
                self.C_ma = C.detach()
            else:
                self.C_ma = alpha * self.C_ma + (1 - alpha) * C.detach()
            C_curr = C.detach().item()  # keep track for logging
            C = C + (self.C_ma - C.detach())  # Move C without affecting its gradient

            # Compute KL divergence
            mu = hgn_output.z_mean
            logvar = hgn_output.z_logvar
            kld = kld_loss(mu=mu, logvar=logvar)

            # normalize by number of frames, channels and pixels per frame
            kld_normalizer = prediction.flatten(1).size(1)
            kld = kld / kld_normalizer

            # Compute losses
            train_loss = kld + self.langrange_multiplier * C

            # clamping the langrange multiplier to avoid inf values
            self.langrange_multiplier = self.langrange_multiplier * torch.exp(
                lagrange_mult_param * C.detach()
            )
            self.langrange_multiplier = torch.clamp(
                self.langrange_multiplier, 1e-10, 1e10
            )

            losses = {
                "loss/train": train_loss.item(),
                "loss/kld": kld.item(),
                "loss/C": C_curr,
                "loss/C_ma": self.C_ma.item(),
                "loss/rec": rec_loss.item(),
                "other/langrange_mult": self.langrange_multiplier.item(),
            }

        else:  # not variational
            # Compute frame reconstruction error
            train_loss = reconstruction_loss(target=target, prediction=prediction)
            losses = {"loss/train": train_loss.item()}

        train_loss.backward()
        self.optimizer.step()

        return losses, hgn_output

    def fit(self):
        """The trainer fits an HGN.

        Returns:
            (HGN) An HGN model that has been fitted to the data
        """

        # Initial values for geco algorithm
        if self.params["networks"]["variational"]:
            self.langrange_multiplier = self.params["geco"][
                "initial_lagrange_multiplier"
            ]
            self.C_ma = None

        # TRAIN
        for ep in range(self.params["optimization"]["epochs"]):
            print(
                "Epoch %s / %s"
                % (str(ep + 1), str(self.params["optimization"]["epochs"]))
            )
            pbar = tqdm.tqdm(self.train_data_loader)
            for batch_idx, rollout_batch in enumerate(pbar):
                # Move to device and change dtype
                rollout_batch = rollout_batch.to(self.device).type(self.dtype)

                # Do an optimization step
                losses, prediction = self.training_step(rollouts=rollout_batch)

                # Log progress
                self.training_logger.step(
                    losses=losses,
                    rollout_batch=rollout_batch,
                    prediction=prediction,
                    model=self.hgn,
                )

                # Progress-bar msg
                msg = ", ".join(
                    [f"{k}: {v:.2e}" for k, v in losses.items() if v is not None]
                )
                pbar.set_description(msg)
            # Save model
            self.hgn.save(self.model_save_file)

        self.test()
        return self.hgn

    def compute_reconst_kld_errors(self, dataloader):
        """Computes reconstruction error and KL divergence.

        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader to retrieve errors from.

        Returns:
            (reconst_error_mean, reconst_error_h), (kld_mean, kld_h): Tuples where the mean and 95%
            confidence interval is shown.
        """
        first = True
        pbar = tqdm.tqdm(dataloader)

        for _, rollout_batch in enumerate(pbar):
            # Move to device and change dtype
            rollout_batch = rollout_batch.to(self.device).type(self.dtype)
            rollout_len = rollout_batch.shape[1]
            input_frames = self.params["optimization"]["input_frames"]
            assert (
                input_frames <= rollout_len
            )  # optimization.use_steps must be smaller (or equal) to rollout.sequence_length
            roll = rollout_batch[:, :input_frames]
            hgn_output = self.hgn.forward(
                rollout_batch=roll, n_steps=rollout_len - input_frames
            )
            target = rollout_batch[
                :, input_frames - 1 :
            ]  # Fit first input_frames and try to predict the last + the next (rollout_len - input_frames)
            prediction = hgn_output.reconstructed_rollout
            error = (
                reconstruction_loss(
                    target=target, prediction=prediction, mean_reduction=False
                )
                .detach()
                .cpu()
                .numpy()
            )
            if self.params["networks"]["variational"]:
                kld = (
                    kld_loss(
                        mu=hgn_output.z_mean,
                        logvar=hgn_output.z_logvar,
                        mean_reduction=False,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                # normalize by number of frames, channels and pixels per frame
                kld_normalizer = prediction.flatten(1).size(1)
                kld = kld / kld_normalizer
            if first:
                first = False
                set_errors = error
                if self.params["networks"]["variational"]:
                    set_klds = kld
            else:
                set_errors = np.concatenate((set_errors, error))
                if self.params["networks"]["variational"]:
                    set_klds = np.concatenate((set_klds, kld))
        err_mean, err_h = mean_confidence_interval(set_errors)
        if self.params["networks"]["variational"]:
            kld_mean, kld_h = mean_confidence_interval(set_klds)
            return (err_mean, err_h), (kld_mean, kld_h)
        else:
            return (err_mean, err_h), None

    def test(self):
        """Test after the training is finished and logs result to tensorboard."""
        print("Calculating final training error...")
        (err_mean, err_h), kld = self.compute_reconst_kld_errors(self.train_data_loader)
        self.training_logger.log_error("Train reconstruction error", err_mean, err_h)
        if kld is not None:
            kld_mean, kld_h = kld
            self.training_logger.log_error("Train KL divergence", kld_mean, kld_h)

        print("Calculating final test error...")
        (err_mean, err_h), kld = self.compute_reconst_kld_errors(self.test_data_loader)
        self.training_logger.log_error("Test reconstruction error", err_mean, err_h)
        if kld is not None:
            kld_mean, kld_h = kld
            self.training_logger.log_error("Test KL divergence", kld_mean, kld_h)


def main(config_file=None):
    if config_file is None:
        with importlib.resources.path(
            hgan.data, "hgn_sample_train_config.yaml"
        ) as config_file:
            with open(config_file, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

        with tempfile.TemporaryDirectory() as temp_dir:
            config["log_dir"] = temp_dir
            config["model_save_dir"] = temp_dir
            trainer = HgnTrainer(config)
            _ = trainer.fit()
    else:
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            trainer = HgnTrainer(config)
            _ = trainer.fit()
