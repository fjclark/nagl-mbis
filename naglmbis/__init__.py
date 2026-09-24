"""
naglmbis
Models built with NAGL to predict MBIS properties.
"""

from . import _version

__version__ = _version.get_versions()["version"]
# make sure all custom features are registered with nagl, which is only installed
# for training
try:
    import nagl.features  # noqa: F401
except ImportError:
    pass
else:
    import naglmbis.features  # noqa: F401
