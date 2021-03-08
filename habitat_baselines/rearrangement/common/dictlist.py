import torch
import numpy


class DictListIter:

    def __init__(self, dictlist):
        self._dictlist = dictlist
        self._children = {key: iter(value) for key, value in dict.items(self._dictlist)}

    def __next__(self):
        return DictList({key: next(child) for key, child in self._children.items()})


class DictList(dict):
    """A dictionnary of lists of same size. Dictionnary items can be
    accessed using `.` notation and list items using `[]` notation.

    Example:
        >>> d = DictList({"a": [[1, 2], [3, 4]], "b": [[5], [6]]})
        >>> d.a
        [[1, 2], [3, 4]]
        >>> d[0]
        DictList({"a": [1, 2], "b": [5]})
    """

    def __init__(self, dict_like=None):
        if dict_like is not None:
            for key, value in (dict(dict_like) if not isinstance(dict_like, dict) else dict_like).items():
                setattr(self, key, value)

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __len__(self):
        return len(next(iter(dict.values(self))))

    def __getitem__(self, index):
        return DictList({key: value[index] for key, value in dict.items(self)})

    def __setitem__(self, index, d):
        for key, value in d.items():
            dict.__getitem__(self, key)[index] = value

    def __iter__(self):
        return DictListIter(self)

    @staticmethod
    def concatenate(dictlists, axis=0):
        if isinstance(dictlists[0], numpy.ndarray):
            return numpy.concatenate(dictlists, axis)
        if isinstance(dictlists[0], torch.Tensor):
            return torch.cat(dictlists, axis)
        for d in dictlists[1:]:
            assert set(d.keys()) == set(dictlists[0].keys())
        return DictList({key: DictList.concatenate([getattr(d, key) for d in dictlists], axis)
                         for key in dictlists[0].keys()})

    @staticmethod
    def repeat(dictlist, repeats, axis):
        if isinstance(dictlist, numpy.ndarray):
            return numpy.repeat(dictlist, repeats, axis)
        if isinstance(dictlist, torch.Tensor):
            torch_repeats = [1] * len(dictlist.shape)
            torch_repeats[axis] = repeats
            return dictlist.repeat(torch_repeats)
        return DictList({key: DictList.repeat(value, repeats, axis)
                         for key, value in dictlist.items()})
