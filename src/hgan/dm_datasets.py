from hgan.dm_hamiltonian_dynamics_suite.hamiltonian_systems import utils

all_systems = (
    "MASS_SPRING",
    "MASS_SPRING_COLORS",
    "MASS_SPRING_COLORS_FRICTION",
    "PENDULUM",
    "PENDULUM_COLORS",
    "PENDULUM_COLORS_FRICTION",
    "DOUBLE_PENDULUM",
    "DOUBLE_PENDULUM_COLORS",
    "DOUBLE_PENDULUM_COLORS_FRICTION",
    "TWO_BODY",
    "TWO_BODY_COLORS",
)


constant_physics = {
    "MASS_SPRING": {
        "k_range": utils.BoxRegion(2.0, 2.0),
        "m_range": utils.BoxRegion(0.5, 0.5),
        "radius_range": utils.BoxRegion(0.1, 1.0),
    },
    "PENDULUM": {
        "m_range": utils.BoxRegion(0.5, 0.5),
        "g_range": utils.BoxRegion(3.0, 3.0),
        "l_range": utils.BoxRegion(1.0, 1.0),
        "radius_range": utils.BoxRegion(1.3, 2.3),
    },
    "DOUBLE_PENDULUM": {
        "m_range": utils.BoxRegion(0.5, 0.5),
        "g_range": utils.BoxRegion(3.0, 3.0),
        "l_range": utils.BoxRegion(1.0, 1.0),
        "radius_range": utils.BoxRegion(1.3, 2.3),
    },
    "TWO_BODY": {
        "m_range": utils.BoxRegion(1.0, 1.0),
        "g_range": utils.BoxRegion(1.0, 1.0),
        "radius_range": utils.BoxRegion(0.5, 1.5),
        "provided_canvas_bounds": utils.BoxRegion(-2.75, 2.75),
    },
}

variable_physics = {
    "MASS_SPRING": {
        "k_range": utils.BoxRegion(2.0, 2.0),
        "m_range": utils.BoxRegion(0.2, 1.0),
        "radius_range": utils.BoxRegion(0.1, 1.0),
    },
    "PENDULUM": {
        "m_range": utils.BoxRegion(0.5, 1.5),
        "g_range": utils.BoxRegion(3.0, 4.0),
        "l_range": utils.BoxRegion(0.5, 1.0),
        "radius_range": utils.BoxRegion(1.3, 2.3),
    },
    "DOUBLE_PENDULUM": {
        "m_range": utils.BoxRegion(0.4, 0.6),
        "g_range": utils.BoxRegion(2.5, 4.0),
        "l_range": utils.BoxRegion(0.75, 1.0),
        "radius_range": utils.BoxRegion(1.0, 2.5),
    },
    "TWO_BODY": {
        "m_range": utils.BoxRegion(0.5, 1.5),
        "g_range": utils.BoxRegion(0.5, 1.5),
        "radius_range": utils.BoxRegion(0.5, 1.5),
        "provided_canvas_bounds": utils.BoxRegion(-5.0, 5.0),
    },
}
