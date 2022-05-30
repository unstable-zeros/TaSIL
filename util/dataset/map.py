from . import Dataset, DatasetIterator
import jax


class MappedDataset(Dataset):
    def __init__(self, dataset, func, rng=None):
        self._dataset = dataset
        self._func = func 
        self._rng = rng

    def __config__(self):
        return {'map': self._func.__config__(), 'data': self._dataset.__config__() }
    
    @property
    def length(self):
        return self._dataset.length

    def get(self, i):
        item  = self._dataset.get(i)
        if item is not None:
            return self._func(item)
        else:
            return None

    def iter(self):
        return MappedIter(self._dataset.iter(), self._func, self._rng)

class MappedIter(DatasetIterator):
    def __init__(self, iter, func, rng):
        self.iter = iter
        self.func = func
        self.rng = rng

    @property
    def has_next(self):
        return self.iter.has_next
    
    def next(self):
        it, x = self.iter.next()
        if self.rng is None:
            x = self.func(x)
            rng = None
        else:
            rng, sk = jax.random.split(self.rng)
            x = self.func(x, rng)
        return MappedIter(it, self.func, rng), x

class FlatMappedDataset(Dataset):
    def __init__(self, dataset, func):
        self._dataset = dataset
        self._func = func
        _, first = self._dataset.iter().next()
        self._factor = len(func(first))

    def __config__(self):
        return {'map': self._func.__config__(), 'data': self._dataset.__config__() }

    @property
    def length(self):
        return self._dataset.length * self._factor

    def iter(self):
        return FlatMappedIter(self._factor, self._dataset.iter(), self._func, [], 0)

class FlatMappedIter(DatasetIterator):
    def __init__(self, factor, iter, func, available, available_idx):
        self.factor = factor
        self.iter = iter
        self.func = func
        self.available = available
        self.available_idx = available_idx

    def __repr__(self):
        return f"flat_map @ {self.iter}"
    
    @property
    def has_next(self):
        return self.available_idx < len(self.available) or self.iter.has_next
    
    def next(self):
        if self.available_idx < len(self.available):
            available = self.available
            idx = self.available_idx
            it = self.iter
        else:
            it, v = self.iter.next()
            # Do the flat mapping
            available = self.func(v)
            idx = 0
        x = available[idx]
        return FlatMappedIter(self.factor, it, self.func, available, idx + 1), x

from util.timer import timed

class ZippedDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        lengths = [ d.length for d in self._datasets ]
        self._len = lengths[0]
    
    @property
    def length(self):
        return self._len

    def iter(self):
        return ZippedIterator([v.iter() for v in self._datasets])

class ZippedIterator(DatasetIterator):
    def __init__(self, iterators):
        self.iterators = iterators

    @property
    def has_next(self):
        return self.iterators[0].has_next
    
    def next(self):
        iterators, values = zip(*[i.next() for i in self.iterators])
        return ZippedIterator(iterators), values

    def skip(self, n):
        iterators = [v.skip(n) for v in self.iterators]
        return ZippedIterator(iterators)
