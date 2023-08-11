import os
import os.path
import tempfile
import logging
from hgan.run import main as run
import hgan.data

logger = logging.getLogger(__name__)


def main():

    existing_ehpr = os.environ.get("HGAN_EXPERIMENT_HAMILTONIAN_PHYSICS_RT")
    existing_espc = os.environ.get("HGAN_EXPERIMENT_SYSTEM_PHYSICS_CONSTANT")

    logger.info("Running model with realtime Hamiltonian Physics")

    os.environ["HGAN_EXPERIMENT_HAMILTONIAN_PHYSICS_RT"] = "1"
    os.environ[
        "HGAN_EXPERIMENT_SYSTEM_PHYSICS_CONSTANT"
    ] = "0"  # Run with varying colors
    with tempfile.TemporaryDirectory() as output_dir:
        data_dir = os.path.dirname(hgan.data.__file__)
        args = [
            "--data_dir",
            data_dir,
            "--generated_videos_dir",
            output_dir,
            "--trained_models_dir",
            output_dir,
            "--video_type",
            "pendulum",
            "--niter",
            "5",
        ]
        run(*args)

    logger.info("Running model against packaged .npz files")

    os.environ["HGAN_EXPERIMENT_HAMILTONIAN_PHYSICS_RT"] = "0"
    with tempfile.TemporaryDirectory() as output_dir:
        data_dir = os.path.dirname(hgan.data.__file__)
        args = [
            "--data_dir",
            data_dir,
            "--generated_videos_dir",
            output_dir,
            "--trained_models_dir",
            output_dir,
            "--video_type",
            "pendulum_colors",
            "--niter",
            "5",
        ]
        run(*args)

    # Reset values
    os.environ["HGAN_EXPERIMENT_HAMILTONIAN_PHYSICS_RT"] = existing_ehpr or ""
    os.environ["HGAN_EXPERIMENT_SYSTEM_PHYSICS_CONSTANT"] = existing_espc or ""
