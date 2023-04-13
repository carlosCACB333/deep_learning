
import os
import sys


def relpath(path):
    """Returns a function that joins a path to the root directory of the project."""

    root_dir = os.path.dirname(os.path.abspath(__file__))

    def join(*paths) -> str:
        return os.path.join(root_dir, path, *paths)
    return join
