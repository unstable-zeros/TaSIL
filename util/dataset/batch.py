from . import Dataset, DatasetIterator
import jax.tree_util as tree_util
import jax.numpy as jnp

class BatchedDataset(Dataset):
    def __init__(self, dataset, batch_size):
        self._dataset = dataset
        self._batch_size = batch_size
        d_len = self._dataset.length
        # Number of batches
        self._len = (d_len + batch_size - 1) // batch_size \
                    if d_len is not None else None

    @property
    def length(self):
        return self._len

    def iter(self):
        return BatchedIter(self._dataset.iter(), self._batch_size)

class BatchedIter(DatasetIterator):
    def __init__(self, it, batch_size):
        self.it = it
        self.batch_size = batch_size

    def __repr__(self):
        return f"batch {self.batch_size} @ {self.it}"
    
    @property
    def has_next(self):
        return self.it.has_next

    def next(self):
        batch = []
        it = self.it
        while len(batch) < self.batch_size and it.has_next:
            it, x = it.next()
            batch.append(x)
        if not batch:
            raise KeyError("No elements left!")
        batch = tree_util.tree_map(lambda *args: jnp.stack(args), *batch)
        return BatchedIter(it, self.batch_size), batch

    def skip(self, n):
        return BatchedIter(self.it.skip(n*self.batch_size), self.batch_size)