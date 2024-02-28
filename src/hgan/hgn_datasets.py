import numpy as np


all_systems_hgn = (
    "mass_spring",
    "pendulum",
    "double_pendulum",
    "two_body",
    "three_body",
)


class BoxRegion:
    def __init__(self, min, max=None):
        self.min = min
        self.max = max
        self.constant = max is None

    def __call__(self):
        if self.constant:
            return self.min
        return np.random.uniform(self.min, self.max)


constant_physics_hgn = {
    "mass_spring": {"elastic_cst": BoxRegion(2.0), "mass": BoxRegion(0.5)},
    "pendulum": {"mass": BoxRegion(0.5), "g": BoxRegion(3.0), "length": BoxRegion(1.0)},
    "double_pendulum": {
        "mass": BoxRegion(1.0),
        "g": BoxRegion(3.0),
        "length": BoxRegion(1.0),
    },
    "two_body": {
        "mass": [BoxRegion(1.0), BoxRegion(1.0)],
        "g": BoxRegion(1.0),
        "orbit_noise": BoxRegion(0.1),
    },
    "three_body": {
        "mass": [BoxRegion(1.0), BoxRegion(1.0), BoxRegion(1.0)],
        "g": BoxRegion(1.0),
        "orbit_noise": BoxRegion(0.1),
    },
}

variable_physics_hgn = {
    "mass_spring": {"elastic_cst": BoxRegion(2.0, 2.0), "mass": BoxRegion(0.2, 1.0)},
    "pendulum": {
        "mass": BoxRegion(0.5, 1.5),
        "g": BoxRegion(3.0, 4.0),
        "length": BoxRegion(0.5, 1.0),
    },
    "double_pendulum": {
        "mass": BoxRegion(0.4, 0.6),
        "g": BoxRegion(2.5, 4.0),
        "length": BoxRegion(0.75, 1.0),
    },
    "two_body": {
        "mass": [BoxRegion(0.5, 1.5), BoxRegion(0.5, 1.5)],
        "g": BoxRegion(0.5, 1.5),
        "orbit_noise": BoxRegion(0.1),
    },
    "three_body": {
        "mass": [BoxRegion(0.5, 1.5), BoxRegion(0.5, 1.5), BoxRegion(0.5, 1.5)],
        "g": BoxRegion(0.5, 1.5),
        "orbit_noise": BoxRegion(0.1),
    },
}
