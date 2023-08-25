from hgan.configuration import config
from hgan.experiment import Experiment


def main():
    experiment = Experiment(config)
    experiment.run()
