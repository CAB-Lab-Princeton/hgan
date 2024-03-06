import os
import os.path
import tempfile
import logging
from hgan.run import main as run
import hgan.data

logger = logging.getLogger(__name__)
this_dir = os.path.dirname(__file__)
data_dir = os.path.dirname(hgan.data.__file__)


def main():

    # Arguments we pass to hgan.run
    # All configuration overrides for the demo are done through environment variables
    args = ["--config-path", this_dir]

    envvars_overridden = (
        "HGAN_EXPERIMENT_SYSTEM_NAME",
        "HGAN_EXPERIMENT_N_EPOCH",
        "HGAN_EXPERIMENT_BATCH_SIZE",
        "HGAN_EXPERIMENT_RT_DATA_GENERATOR",
        "HGAN_EXPERIMENT_SYSTEM_PHYSICS_CONSTANT",
        "HGAN_EXPERIMENT_NDIM_LABEL",
        "HGAN_EXPERIMENT_NDIM_PHYSICS",
        "HGAN_PATHS_INPUT",
        "HGAN_PATHS_OUTPUT",
    )

    envvars = {k: os.environ.get(k) for k in envvars_overridden}

    with tempfile.TemporaryDirectory() as output_dir:
        os.environ["HGAN_EXPERIMENT_N_EPOCH"] = "5"
        os.environ["HGAN_EXPERIMENT_BATCH_SIZE"] = "4"
        os.environ["HGAN_PATHS_OUTPUT"] = output_dir

        logger.info("Running model with realtime Hamiltonian Physics")

        os.environ["HGAN_EXPERIMENT_RT_DATA_GENERATOR"] = "hgn"
        os.environ[
            "HGAN_EXPERIMENT_SYSTEM_PHYSICS_CONSTANT"
        ] = "0"  # Run with varying colors
        run(*args)

    with tempfile.TemporaryDirectory() as output_dir:
        os.environ["HGAN_EXPERIMENT_N_EPOCH"] = "5"
        os.environ["HGAN_EXPERIMENT_BATCH_SIZE"] = "4"
        os.environ["HGAN_PATHS_OUTPUT"] = output_dir

        logger.info("Running model against packaged .npz files")

        os.environ["HGAN_EXPERIMENT_RT_DATA_GENERATOR"] = ""
        os.environ["HGAN_EXPERIMENT_NDIM_LABEL"] = "0"
        os.environ["HGAN_EXPERIMENT_NDIM_PHYSICS"] = "0"
        os.environ["HGAN_EXPERIMENT_SYSTEM_NAME"] = "pendulum_colors"
        os.environ["HGAN_PATHS_INPUT"] = data_dir
        run(*args)

    # Reset envvars
    for k, v in envvars.items():
        if v is not None:
            os.environ[k] = v
