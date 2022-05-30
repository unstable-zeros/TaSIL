from . import Dataset, DatasetIterator
import jax

class ShuffledDataset(Dataset):
    def __init__(self, dataset, rng, buffer_size=None):
        self._dataset = dataset
        self._buffer_size = buffer_size
        self._rng = rng
    
    @property
    def length(self):
        return self._dataset.length

    def iter(self):
        if self._buffer_size is None:
            indices = jax.random.permutation(self._rng, len(self._dataset))
            return PermutationIterator(self._dataset, indices, 0, len(self._dataset))
        else:
            return ShuffledIterator(self._dataset.iter(), self._rng, [], self._buffer_size)

class PermutationIterator(DatasetIterator):
    def __init__(self, dataset, indices, off, length):
        self.dataset = dataset
        self.indices = indices
        self.off = off
        self.len = length

    @property
    def has_next(self):
        return self.off < self.len

    def next(self):
        idx = self.indices[self.off]
        x = self.dataset.get(idx)
        return PermutationIterator(self, self.dataset, self.indices, self.off + 1, self.len)

class ShuffledIterator(DatasetIterator):
    def __init__(self, it, key, buffer, buffer_size):
        self.it = it
        self.key = key
        self.buffer = buffer
        self.buffer_size = buffer_size
    
    @property
    def has_next(self):
        return len(self.buffer) > 0 or self.it.has_next
    
    def next(self):
        buffer = list(self.buffer)
        it = self.it
        # Fill the buffer as much as possible. If buffer_size is None
        # we read in the whole dataset
        while ((len(buffer) < self.buffer_size) if self.buffer_size is not None else True) and it.has_next:
            it, x = it.next()
            buffer.append(x)
        if buffer:
            key, sk = jax.random.split(self.key)
            idx = jax.random.randint(sk, (), 0, len(buffer))
            x = buffer.pop(idx)
        else:
            key = self.key
            x = None
        return ShuffledIterator(it, key, buffer, self.buffer_size), x
    
    def skip(self, n):
        raise NotImplementedError("TODO: Cannot skip on a shuffled iterator")