from . import Dataset, DatasetIterator

import pickle
import json
import numpy as np
import jax.numpy as jnp
import flax
import os
from os import path
from jax.tree_util import tree_map

def serialize_config(config):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, flax.core.frozen_dict.FrozenDict):
                return obj.unfreeze()
            if isinstance(obj, jnp.DeviceArray):
                return np.array(obj).tolist()
            return json.JSONEncoder.default(self, obj)
    config = json.dumps(config, cls=NumpyEncoder)
    return config

class DatasetWriter:
    def __init__(self, path):
        self.path = path
        self.count = 0
        self.buffer = []
        self.current_fn = None
        self.current_file = None

    def _open_tmp(self):
        if self.current_fn is None:
            self.current_fn = 'tmp'
            self.current_file = open(path.join(self.path, self.current_fn), 'wb')

    def _close_tmp(self):
        if self.current_fn is not None:
            self.current_file.flush()
            self.current_file.close()
            current_path = path.join(self.path, self.current_fn)
            new_path = path.join(self.path, f'{self.count}.pk')
            os.rename(current_path, new_path)
            self.current_fn = None
            self.current_file = None
    
    def write(self, item):
        self._open_tmp()
        self.buffer.append(item)
        self.count = self.count + 1
        if self.count % 1000 == 0:
            # Write the buffer to disk
            buffer = tree_map(lambda *args: jnp.stack(args), *self.buffer)
            pickle.dump(buffer, self.current_file)
            self.buffer = []
            self._close_tmp()

    def close(self):
        if self.buffer:
            buffer = tree_map(lambda *args: jnp.stack(args), *self.buffer)
            pickle.dump(buffer, self.current_file)
        self._close_tmp()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

class SavedDataset(Dataset):
    def __init__(self, path):
        files = [f for f in os.listdir(path) if f.endswith(".pk")]
        offsets = [int(f.rsplit(".", 1)[0]) for f in files]
        self._length = max(offsets)
        self._interval = offsets[0]
        self._paths = [os.path.join(path, f) for f in files]

        self._tmp_idx = None
        self._tmp_loaded = None

    def _load(self, idx):
        if self._tmp_idx == idx:
            return self._tmp_loaded
        else:
            path = self._paths[idx]
            with open(path, 'rb') as f:
                d = pickle.load(f)
                self._tmp_idx = idx
                self._tmp_loaded = d
        return self._tmp_loaded

    @property
    def length(self):
        return self._length

    def get(self, i):
        if i > self._length:
            raise KeyError("Out of range")
        idx = i // self._interval
        offset = i % self._interval
        data = self._load(idx)
        sample = tree_map(lambda x: x[offset], data)
        return sample