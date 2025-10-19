"""Record linkage package wrapping the standalone GPU pipeline."""

from importlib import resources
from pathlib import Path


def package_path() -> Path:
    """Return the filesystem path to the bundled record linkage assets."""
    return Path(resources.files(__package__))


__all__ = ["package_path"]
