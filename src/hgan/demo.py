import os.path
import tempfile
from hgan.run import main as run
import hgan.data


def main():
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
