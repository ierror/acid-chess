import os
import pathlib
from contextlib import contextmanager


@contextmanager
def posix_path_compatibility():
    posix_backup = pathlib.PosixPath
    try:
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath
        yield
    finally:
        pathlib.PosixPath = posix_backup
