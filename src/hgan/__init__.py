# Having both tensorflow/torch in requirements has been known to cause import issues
# with cudnn shared libraries etc., unless we first import them in this order:
import torch  # noqa: F401
import tensorflow  # noqa: F401

# The _version.py file is managed by setuptools-scm
#   and is not in version control.
try:
    from hgan._version import version as __version__  # type: ignore
except ModuleNotFoundError:
    # We're likely running as a source package without installation
    __version__ = "vSource"


import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO,
)
