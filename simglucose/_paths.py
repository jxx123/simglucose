"""Centralized path resolution for simglucose data files.

Uses __file__-relative paths so editable installs work correctly.
"""

import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts: str) -> str:
    """Resolve a path relative to the simglucose package root.

    Usage:
        resource_path("params", "vpatient_params.csv")
    """
    return os.path.join(_PACKAGE_DIR, *parts)
