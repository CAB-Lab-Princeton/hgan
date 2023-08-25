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
        "HGAN_EXPERIMENT_HAMILTONIAN_PHYSICS_RT",
        "HGAN_EXPERIMENT_SYSTEM_PHYSICS_CONSTANT",
        "HGAN_PATHS_INPUT",
        "HGAN_PATHS_OUTPUT",
    )

    envvars = {k: os.environ.get(k) for k in envvars_overridden}

    with tempfile.TemporaryDirectory() as output_dir:
        os.environ["HGAN_PATHS_OUTPUT"] = output_dir

        logger.info("Running model with realtime Hamiltonian Physics")

        os.environ["HGAN_EXPERIMENT_HAMILTONIAN_PHYSICS_RT"] = "1"
        os.environ[
            "HGAN_EXPERIMENT_SYSTEM_PHYSICS_CONSTANT"
        ] = "0"  # Run with varying colors
        run(*args)

        logger.info("Running model against packaged .npz files")

        os.environ["HGAN_EXPERIMENT_HAMILTONIAN_PHYSICS_RT"] = "0"
        os.environ["HGAN_EXPERIMENT_SYSTEM_NAME"] = "pendulum_colors"
        os.environ["HGAN_PATHS_INPUT"] = data_dir
        run(*args)

    # Reset envvars
    for k, v in envvars.items():
        os.environ[k] = v or ""
