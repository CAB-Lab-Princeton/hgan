import os
from hgan.configuration import config
from hgan.experiment import Experiment


def test_experiment():

    old_value = os.environ.get("HGAN_ARCH_DE")
    os.environ["HGAN_ARCH_DE"] = "5432"

    experiment = Experiment(config)

    # Certain derived configuration options can be accessed from the Experiment object
    assert experiment.d_E == 5432
    # Configuration options can still be accessed by reaching in to the 'config' attribute first
    assert experiment.config.arch.de == 5432

    if old_value is not None:
        os.environ["HGAN_ARCH_DE"] = old_value
