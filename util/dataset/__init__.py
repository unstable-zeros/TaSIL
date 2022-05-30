from loguru import logger
import threading
import traceback
import sys
import queue
import hashlib
import tqdm
import os
import multiprocessing
import jax.experimental.host_callback
import jax.tree_util as tree_util
import jax.numpy as jnp

from util.logging import logging_redirect_tqdm

INFINITE = float("inf")

# The host stream mapping
_JAX_HOSTS = {}
_JAX_NEXT_HOST_ID = 0

class Dataset:
    # Can return None for either unknown or infinite length
    @property
    def length(self):
        raise NotImplementedError(f"length not implemented for {self.__class__.__name__}")

    @property
    def schema(self):
        raise NotImplementedError(f"schema not implemented for {self.__class__.__name__}")

    # You *MUST* either override get() or iter()
    # return None if out of bounds
    def get(self, i):
        raise NotImplementedError(f"get() not implemented for {self.__class__.__name__}")

    def iter(self):
        return SimpleIterator(self, 0)

    def __config__(self):
        raise NotImplementedError(f"__config__ not implemented for {self.__class__.__name__}")

    # DO NOT OVERRIDE BELOW HERE....
    def __len__(self):
        length = self.length
        if length == INFINITE:
            raise TypeError("Infinite length")
        else:
            return length

    def __getitem__(self, i):
        if isinstance(i, slice):
            return self.slice(i.start, i.stop, i.step)
        return self.get(i)

    # Builtin transformations (Do not override)
    def slice(self, start, stop, step=None):
        from .slice import SlicedDataset
        return SlicedDataset(self, start, stop, step)

    def until(self, n):
        return self.slice(0, n)
    
    def batch(self, n):
        from .batch import BatchedDataset
        return BatchedDataset(self, n)

    def shuffle(self, seed, buffer_size=None):
        from .shuffle import ShuffledDataset
        return ShuffledDataset(self, seed, buffer_size)

    def map(self, fun, rng=None):
        if fun is None:
            return self
        from .map import MappedDataset
        return MappedDataset(self, fun, rng)

    def flat_map(self, fun):
        if fun is None:
            return self
        from .map import FlatMappedDataset
        return FlatMappedDataset(self, fun)

    def cache(self, name, progress=True):
        # Prevent cyclical imports
        from .io import serialize_config, SavedDataset, DatasetWriter
        config = serialize_config(self.__config__())
        hash = hashlib.sha256(config.encode('utf-8')).hexdigest()
        filename = f'{name}_{hash}_data'
        path = os.path.join('.datasets', f"{filename}")
        if os.path.exists(path):
            logger.info(f'Using cached {filename}...')
            return SavedDataset(path)

        logger.info(f'Caching data...')
        os.makedirs(path, exist_ok=True)
        with DatasetWriter(path) as writer:
            with tqdm.tqdm(total=len(self)) as pbar:
                with logging_redirect_tqdm():
                    iterator = self.iter()
                    while iterator.has_next:
                        iterator, sample = iterator.next()
                        writer.write(sample)
                        pbar.update(1)

        logger.info(f'Dataset cached in {filename}')
        return SavedDataset(path)

    def preload(self, progress=True):
        iterator = self.iter()
        data = []
        with tqdm.tqdm(total=len(self)) as pbar:
            with logging_redirect_tqdm():
                while iterator.has_next:
                    iterator, sample = iterator.next()
                    data.append(sample)
                    pbar.update(1)
        data = tree_util.tree_map(lambda *args: jnp.stack(args), *data)
        return Dataset.from_pytree(data)

    @staticmethod
    def zip(*args):
        from .map import ZippedDataset
        return ZippedDataset(args)

    @staticmethod
    def join(a, b):
        data = jax.tree_util.tree_map(lambda *args: jnp.concatenate(args), a.data, b.data)
        return PyTreeDataset(data)

    @staticmethod
    def from_pytree(data):
        return PyTreeDataset(data)

# Dataset iterators should be immutable!
class DatasetIterator:
    @property
    def has_next(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement has_next")

    def next(self):
        raise NotImplementedError(f"{self.__class__.__name__} must implement next")

    def skip(self, n):
        it = self
        for i in range(n):
            it, _ = it.next()
        return it

    # Create a cpu --> jax loader
    def stream(self, prefetch=16):
        _, example = self.next()
        host_id = JaxLoaderHost.make(self, prefetch)
        return JaxLoader(jnp.array(host_id), 0, self.has_next, example)

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class SimpleIterator(DatasetIterator):
    def __init__(self, dataset, i):
        self.dataset = dataset
        self.i = i
    
    @property
    def has_next(self):
        return self.i < self.dataset.length

    def next(self):
        return SimpleIterator(self.dataset, self.i + 1), self.dataset.get(self.i)
    
    def skip(self, n):
        return SimpleIterator(self.dataset, self.i + n)

    def tree_flatten(self):
        return ((self.dataset, self.i), None)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
class PyTreeDataset(Dataset):
    def __init__(self, data, num=None):
        self.data = data
        if num is None:
            nums_tree = tree_util.tree_map(lambda x: jnp.shape(x)[0], self.data)
            all_nums = tree_util.tree_flatten(nums_tree)[0]
            num = all_nums[0]
            self.num = num
        else:
            self.num = num

    @property
    def length(self):
        return self.num

    def get(self, i):
        return tree_util.tree_map(lambda x: x[i], self.data)
    
    def shuffle(self, seed, buffer_size=None):
        perm = jax.random.permutation(seed, self.length)
        shuffled_data = jax.tree_util.tree_map(lambda x: x[perm], self.data)
        return PyTreeDataset(shuffled_data, self.num)

    def batch(self, n):
        def make_batch(x):
            x = x.reshape((n, -1,) + x.shape[1:])
            x = jnp.swapaxes(x, 0, 1)
            return x
        batch_data = tree_util.tree_map(
            make_batch, self.data
        )
        return PyTreeDataset(batch_data)

    def preload(self, progress=True):
        return self

    def tree_flatten(self):
        return ((self.data, self.num), None)

    @staticmethod
    def tree_unflatten(aux_data, children):
        return PyTreeDataset(*children)

class JaxLoaderHost:
    def __init__(self, iterator, prefetch):
        self._iterator = iterator
        self._thread = threading.Thread(target=self.fill)
        self._take_requests = queue.Queue()
        self._take_queue = queue.Queue()
        self._thread.start()
        for i in range(prefetch):
            self._take_requests.put((None,))
    
    def fill(self):
        try:
            while True:
                _ = self._take_requests.get()
                for _ in range(32):
                    new_it, batch = self._iterator.next()
                    has_next = new_it.has_next
                    self._iterator = new_it
                    self._take_queue.put((batch, has_next))
                    if not has_next:
                        break
                if not has_next:
                    break
        except:
            print('Error in dataloader:')
            traceback.print_exc()
            sys.exit(1)

    def take(self, i):
        try:
            self._take_requests.put((None,))
            return self._take_queue.get_nowait()
        except queue.Empty:
            logger.debug("Stalling for data...")
            return self._take_queue.get()

    @staticmethod
    def make(iterator, prefetch):
        host = JaxLoaderHost(iterator, prefetch)
        global _JAX_NEXT_HOST_ID
        id = _JAX_NEXT_HOST_ID
        _JAX_NEXT_HOST_ID = _JAX_NEXT_HOST_ID + 1
        _JAX_HOSTS[id] = host
        return id

def _loader_host_fn(args):
    host_id, index = args
    host = _JAX_HOSTS[host_id.item()]
    sample, has_next = host.take(index)
    return sample, has_next

@register_pytree_node_class
class JaxLoader(DatasetIterator):
    def __init__(self, host, index, has_next, example):
        self._host = host
        self._index = index
        self._has_next = has_next
        self._example = example

    @property
    def has_next(self):
        return self._has_next

    @jax.jit
    def next(self):
        batch, new_has_next = jax.experimental.host_callback.call(_loader_host_fn, (self._host, self._index),
                                                                    result_shape=(self._example, bool))
        return JaxLoader(self._host, self._index + 1, new_has_next, self._example), batch

    def tree_flatten(self):
        children = (self._index, self._has_next, self._host, self._example)
        return children, None
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        index, has_next, host, example = children
        return cls(host, index, has_next, example)