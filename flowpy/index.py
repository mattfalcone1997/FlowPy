import itertools
import warnings
import numpy as np
import numbers
import copy

from pandas._libs.index import ObjectEngine
import pandas as pd
from pandas.core.indexes.multi import MultiIndexPyIntEngine, MultiIndexUIntEngine

from abc import abstractmethod, abstractproperty, ABC
from .io import hdfHandler
from ._api import rcParams
from .utils import find_stack_level
from typing import Iterable


class IndexBase(ABC):
    @abstractproperty
    def is_MultiIndex(self):
        pass

    @abstractmethod
    def _check_update(self):
        pass

    def get_loc(self, key):
        self._check_update()
        return self._index.get_loc(self._item_handler(key))

    @classmethod
    def _item_handler(cls, item):
        if isinstance(item, tuple):
            return tuple(cls._item_handler(k) for k in item)
        if isinstance(item, numbers.Number):
            return "%.9g" % item
        elif isinstance(item, str):
            return "%.9g" % float(item) if item.isnumeric() else item
        else:
            return str(item)

    @abstractmethod
    def to_array(self, string=True):
        pass

    def __contains__(self, key):
        key = self._item_handler(key)
        return key in self._indices

    def remove(self, key):
        if key not in self:
            msg = f"Key {key} not present in indexer"
            raise KeyError(msg)

        self._indices.remove(key)
        self._updated = False

    @abstractmethod
    def get_inner_index(self):
        pass

    def update_key(self, old_key, new_key):
        old_key = self._item_handler(old_key)
        new_key = self._item_handler(new_key)

        if old_key not in self:
            msg = f"The key {old_key} must be an existing key in the indexer"
            raise KeyError(msg)

        i = self._indices.index(old_key)
        self._indices[i] = new_key
        self._updated = False

    def get_index(self):
        return self._indices

    def __iter__(self):
        return self._indices.__iter__()

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, key):
        return self._indices.__getitem__(key)

    def __repr__(self):
        name = self.__class__.__name__
        return "%s(%s)" % (name, self._indices)

    def __str__(self):
        name = self.__class__.__name__
        return "%s(%s)" % (name, self._indices)

    def __eq__(self, other_index):
        return all(x == y for x, y in zip(self, other_index))

    def copy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        return self.__class__(self._indices)

    @staticmethod
    def is_listkey(key):
        if isinstance(key, tuple):
            if any([isinstance(k, (list, Index)) for k in key]):
                return True
        elif isinstance(key, (list, Index)):
            return True
        return False

    @staticmethod
    def is_multikey(key):
        if isinstance(key, tuple):
            if len(key) > 1:
                return True
        return False

    @staticmethod
    def _getitem_process_list(key):
        if isinstance(key, tuple):
            if not isinstance(key[0], (list, Index)):
                inner_key = [key[0]]
            else:
                inner_key = list(key[0])
            if not isinstance(key[1], (list, Index)):
                outer_key = [key[1]]
            else:
                outer_key = list(key[1])

            key_list = list(itertools.product(inner_key, outer_key))
        else:
            if not isinstance(key, list):
                msg = "This function should only be called on keys containing lists"
                raise TypeError(msg)
            key_list = key

        return key_list

    def check_inner(self, key, err):
        if key not in self.get_inner_index():
            raise err from None

        return key

    def _check_indices(self, indices):
        indices = list(indices)

        def _is_unique(lst):
            seen = set()
            return not any(i in seen or seen.add(i) for i in lst)

        if not _is_unique(indices) and len(indices) > 1:
            msg = "Indices must be unique"
            raise ValueError(msg)

        return indices

    def extend(self, other_index):
        if type(other_index) != self.__class__:
            msg = "The type of merging index must be the same as the current index"
            raise TypeError(msg)

        for key in other_index:
            self.append(key)
        self._updated = False

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        obj = self.__class__(d)
        self.__dict__ = obj.__dict__


class Index(IndexBase):
    def __init__(self, indices: Iterable):

        if not all(isinstance(i, str) for i in indices):
            msg = "All elements of indices must be strings"
            raise TypeError(msg)

        self._indices = list(indices)
        self._updated = False
        self._check_update()

        # self._update_internals()

    # def to_hdf(self,hdf_obj, name):
    #     if not all(isinstance(x,self[0]) for x in self):
    #         raise TypeError("Data can only be saved to hdf if"
    #                         " all components in index are the same")

    #     grp = hdf_obj.create_group(name)
    #     grp.create_dataset('indices',data=self._data)

    def _check_update(self):
        if not self._updated:
            self._index = pd.Index(self._indices)
            self._updated = True

    def to_array(self, string=True):
        self._check_update()
        type = np.string_ if string else object
        array = []
        for index in self:
            array.append(type(index))
        return np.array(array)

    def get_inner_index(self):
        return self.get_index()

    @property
    def is_MultiIndex(self):
        return False

    def append(self, key):
        key = self._item_handler(key)
        self._indices.append(key)
        self._updated = False


class MultiIndex(IndexBase):
    def __init__(self, indices):
        if not all(isinstance(index, tuple) for index in indices):
            msg = "All indices must be tuples"
            raise TypeError(msg)

        if not all(all(isinstance(i, str) for i in index) for index in indices):
            msg = "All elements for the tuples must be strings"
            raise TypeError(msg)

        self._indices = list(indices)
        self._updated = False
        self._check_update()

    def _check_update(self):
        if not self._updated:
            self._index = pd.MultiIndex.from_tuples(self._indices)
            self._updated = True
        # self._update_internals()

    # def to_hdf(self,hdf_obj, name):
    #     if not all(isinstance(x,self[0]) for x in self):
    #         raise TypeError("Data can only be saved to hdf if"
    #                         " all components in index are the same")

    #     grp = hdf_obj.create_group(name)

    #     grp.create_dataset('inner_index',data=self.inner_index)
    #     grp.create_dataset('outer_index',data=self.outer_index)

    # @classmethod
    # def from_hdf(cls,hdf_obj, name):
    #     if 'inner_index' in hdf_obj[name].keys():
    #         pass
    #     else:
    #         index_array = hdf_obj.contruct_index_from_hdfkey('index')
    #         return cls(index_array)
    # def _update_internals(self):
    #     self._outer_index = Index(set([x[0] for x in self.get_index()]))
    #     self._inner_index = Index(set([x[1] for x in self.get_index()]))

    #     self.__engine = self._create_mapping()

    def update_inner_key(self, old_key, new_key):
        old_key = self._item_handler(old_key)
        new_key = self._item_handler(new_key)
        for i, key in enumerate(self._indices):
            if old_key == key[1]:
                self._indices[i] = (key[0], new_key)

        self._updated = False

    def update_outer_key(self, old_key, new_key):
        old_key = self._item_handler(old_key)
        new_key = self._item_handler(new_key)
        for i, key in enumerate(self._indices):
            if old_key == key[0]:
                self._indices[i] = (new_key, key[1])

        self._updated = False

    # def __contains__(self, key):
    #     return self._index.__contains__(key)
    # @property
    # def _levels(self):
    #     return [self.get_outer_index(),self.get_inner_index()]

    # @property
    # def _codes(self):
    #     outer_level, inner_level = self._levels

    #     outer_code_map = dict(zip(outer_level,range(len(outer_level))))
    #     inner_code_map = dict(zip(inner_level,range(len(inner_level))))

    #     inner_index = [x[1] for x in self.get_index()]
    #     outer_index = [x[0] for x in self.get_index()]

    #     outer_code = [outer_code_map[x] for x in outer_index]
    #     inner_code = [inner_code_map[x] for x in inner_index]

    #     return [outer_code, inner_code]

    @property
    def outer_index(self):
        return list(set([x[0] for x in self._indices]))

    @property
    def inner_index(self):
        return list(set([x[1] for x in self._indices]))

    def to_array(self, string=True):

        type = np.string_ if string else object
        array = []
        for index in self:
            array.append(np.array(index, dtype=type))
        return np.array(array)

    def check_outer(self, key, err, warn=None):
        key = self._item_handler(key)
        outer_index = self.get_outer_index()
        if key not in outer_index:
            if len(outer_index) > 1:
                raise err from None
            else:
                if key != 'None' and warn is not None:
                    warnings.warn(warn, stacklevel=find_stack_level())
                key = self.get_outer_index()[0]

        return key

    def get_inner_index(self):
        return self.inner_index

    def get_outer_index(self):
        return self.outer_index

    def append(self, key):
        if len(key) != 2:
            raise ValueError("key length is wrong")
        self._indices.append((self._item_handler(key[0]),
                              self._item_handler(key[1])))
        self._updated = False

    # def _create_mapping(self):
    #     sizes = np.ceil(np.log2([len(level) + 1 for level in self._levels]))

    #     lev_bits = np.cumsum(sizes[::-1])[::-1]
    #     offsets = np.concatenate([lev_bits[1:], [0]]).astype("uint64")

    #     if lev_bits[0] > 64:
    #         return MultiIndexPyIntEngine(self._levels, self._codes, offsets)
    #     return MultiIndexUIntEngine(self._levels, self._codes, offsets)

    # @property
    # def _mapping(self):
    #     return self.__engine

    @property
    def is_MultiIndex(self):
        return True


class structIndexer:

    def __new__(cls, index):
        index = cls.index_constructor(index)
        if all(isinstance(ind, tuple) for ind in index):
            return MultiIndex(index)
        else:
            return Index(index)

    @staticmethod
    def _construct_arrays_check(index):
        if len(index) != 2:
            return False

        allowed_types = (list, Index, np.ndarray)
        if not all(isinstance(ind, allowed_types) for ind in index):
            return False

        if len(index[0]) != len(index[1]):
            msg = "Invalid iterable used for indexing"
            raise TypeError(msg)

        return True

    @staticmethod
    def _construct_tuples_check(index):
        if not all(isinstance(ind, tuple) for ind in index):
            return False

        if not all(len(x) == 2 for x in index):
            msg = "The length of each tuple must be 2"
            raise ValueError(msg)

        return True

    @staticmethod
    def _construct_1D_check(index):
        return all(isinstance(ind, (numbers.Number, str)) for ind in index)

    @classmethod
    def index_constructor(cls, index):
        # two options for index construction
        # list with two list => [outer_index, inner_index]
        # list of tuples => each tuple an index in the data struct
        if cls._construct_arrays_check(index):

            outer_index = list(IndexBase._item_handler(k) for k in index[0])
            inner_index = list(IndexBase._item_handler(k) for k in index[1])
            index = list(zip(outer_index, inner_index))

        elif cls._construct_tuples_check(index):
            inner_index = list(IndexBase._item_handler(k[1]) for k in index)
            outer_index = list(IndexBase._item_handler(k[0]) for k in index)
            index = list(zip(outer_index, inner_index))

        elif cls._construct_1D_check(index):
            index = list(IndexBase._item_handler(k) for k in index)
        else:
            msg = "The index passed is invalid"
            raise ValueError(msg)

        return index
