import importlib.resources
from pathlib import Path


def load_data(filename):
    p_filename = Path(filename)
    if p_filename.suffix == '.npz':
        import numpy as np
        my_resources = importlib.resources.files(__name__)
        with my_resources.joinpath(filename).open() as stream:
            return np.load(stream)
    else:
        from ase.io import iread
        my_resources = importlib.resources.files(__name__)
        with my_resources.joinpath(filename).open() as f:
            return iread(f, ':')
