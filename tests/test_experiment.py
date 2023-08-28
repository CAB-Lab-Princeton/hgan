import os
from hgan.configuration import config
from hgan.experiment import Experiment


def test_experiment():

    old_value = os.environ.get("HGAN_EXPERIMENT_NDIM_EPSILON")
    os.environ["HGAN_EXPERIMENT_NDIM_EPSILON"] = "5432"

    experiment = Experiment(config)

    # Certain derived configuration options can be accessed from the Experiment object
    assert experiment.ndim_epsilon == 5432
    # Configuration options can still be accessed by reaching in to the 'config' attribute first
    assert experiment.config.experiment.ndim_epsilon == 5432

    if old_value is not None:
        os.environ["HGAN_EXPERIMENT_NDIM_EPSILON"] = old_value
