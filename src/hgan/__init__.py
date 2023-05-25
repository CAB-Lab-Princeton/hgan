# The _version.py file is managed by setuptools-scm
#   and is not in version control.
try:
    from hgan._version import version as __version__  # type: ignore
except ModuleNotFoundError:
    # We're likely running as a source package without installation
    __version__ = "vSource"
