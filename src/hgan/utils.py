import os
import time
import math
from typing import Any, Mapping, Callable, Tuple
import tensorflow as tf
import torch
import numpy as np


def setup_reproducibility(seed):
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)


def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2) * 24))
    h = math.floor(s / (60**2)) - d * 24
    m = math.floor(s / 60) - h * 60 - d * 24 * 60
    s = s - m * 60 - h * (60**2) - d * 24 * (60**2)
    return "%dd %dh %dm %ds" % (d, h, m, s)


# for input noises to generate fake video
# note that noises are trimmed randomly from n_frames to T for efficiency
def trim_noise(noise, T):
    start = np.random.randint(0, noise.size(1) - T + 1)
    end = start + T
    return noise[:, start:end, :, :, :]


def choose_nonlinearity(name):
    return {
        "tanh": torch.nn.Tanh(),
        "relu": torch.nn.ReLU(),
        "sigmoid": torch.sigmoid,
        "softplus": torch.nn.Softplus(),
        "leakyrelu": torch.nn.LeakyReLU(),
        "selu": torch.nn.functional.selu,
        "elu": torch.nn.functional.elu,
        "swish": lambda x: x * torch.sigmoid(x),
    }[name]


# for true video
def trim(video, T=16):
    start = np.random.randint(0, video.shape[1] - (T + 1))
    end = start + T
    return video[:, start:end, :, :]


def load_filenames_and_parse_fn(
    path: str, tfrecord_prefix: str
) -> Tuple[Tuple[str], Callable[[str], Mapping[str, Any]]]:
    """Returns the file names and read_fn based on the number of shards."""
    file_name = os.path.join(path, f"{tfrecord_prefix}.tfrecord")
    if not os.path.exists(file_name):
        raise ValueError(f"The dataset file {file_name} does not exist.")
    features_file = os.path.join(path, "features.txt")
    if not os.path.exists(features_file):
        raise ValueError(
            f"The dataset features file {features_file} does not " f"exist."
        )
    with open(features_file, "r") as f:
        dtype_dict = dict()
        shapes_dict = dict()
        parsing_description = dict()
        for line in f:
            key = line.split(", ")[0]
            shape_string = line.split("(")[1].split(")")[0]
            shapes_dict[key] = tuple(int(s) for s in shape_string.split(",") if s)
            dtype_dict[key] = line.split(", ")[-1][:-1]
            if dtype_dict[key] == "uint8":
                parsing_description[key] = tf.io.FixedLenFeature([], tf.string)
            elif dtype_dict[key] in ("float32", "float64"):
                parsing_description[key] = tf.io.VarLenFeature(tf.int64)
            else:
                parsing_description[key] = tf.io.VarLenFeature(dtype_dict[key])

    def parse_fn(example_proto: str) -> Mapping[str, Any]:
        raw = tf.io.parse_single_example(example_proto, parsing_description)
        parsed = dict()
        for name, dtype in dtype_dict.items():
            value = raw[name]
            if dtype == "uint8":
                value = tf.image.decode_png(value)
            else:
                value = tf.sparse.to_dense(value)
                if dtype in ("float32", "float64"):
                    value = tf.bitcast(value, type=dtype)
            value = tf.reshape(value, shapes_dict[name])
            if "/" in name:
                k1, k2 = name.split("/")
                if k1 not in parsed:
                    parsed[k1] = dict()
                parsed[k1][k2] = value
            else:
                parsed[name] = value
        return parsed

    return (file_name,), parse_fn
