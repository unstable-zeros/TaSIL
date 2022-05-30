from . import Dataset, DatasetIterator

class SlicedDataset(Dataset):
    def __init__(self, dataset, start, stop, step):
        start, step = start or 0, step or 1
        d_len =  dataset.length
        stop = min(d_len, stop) if stop is not None and d_len is not None else stop

        self.num = None if stop is None else (stop - start) // step
        self.dataset = dataset
        self.start = start
        self.stop = stop
        self.step = step

    def __config__(self):
        return { 'sliced': self.dataset.__config__(), 
                 'start': self.start, 'stop': self.stop,
                 'step': self.step }

    @property
    def length(self):
        return self.num
    
    def get(self, i):
        if i > self.num:
            raise IndexError("Out of dataset range")
        return self.dataset.get(self.start + self.step*i)
    
    def iter(self):
        sub_iter = self.dataset.iter()
        if self.start > 0:
            sub_iter = sub_iter.skip(self.start)
        remaining = None if self.stop is None else (self.stop - self.start) // self.step
        return SlicedIterator(sub_iter, self.step, remaining)

    # Will recompute a new slice rather
    # than chaining sliced datasets for efficiency
    def slice(self, start, stop, step):
        mapped_stop = stop*self.step if stop is not None else None
        new_stop = (mapped_stop if self.stop is None else
                   self.stop if mapped_stop is None else min(mapped_stop, self.stop))
        return self.dataset.slice(
            self.start + start*self.step,
            new_stop,
            step*self.step)

class SlicedIterator(DatasetIterator):
    def __init__(self, sub_iter, step, remaining):
        self.sub_iter = sub_iter
        self.step = step
        self.remaining = remaining

    @property
    def has_next(self):
        return (self.remaining > 0 if self.remaining is not None else True) \
                and self.sub_iter.has_next
    
    def next(self):
        if self.remaining is not None:
            if self.remaining <= 0:
                raise StopIteration()
            remaining = self.remaining - 1
        else:
            remaining = None

        sub_iter, x = self.sub_iter.next()
        if self.step > 1:
            sub_iter = sub_iter.skip(self.step - 1)
        return SlicedIterator(sub_iter, self.step, remaining), x
    
    def skip(self, n):
        sub_iter = self.sub_iter.skip(self.step*n)
        return SlicedIterator(sub_iter, self.step, self.remaining - n if self.remaining is not None else None)