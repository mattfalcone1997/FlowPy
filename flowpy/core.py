"""
# CHAPSim_dtypes
A module for the CHAPSim_post postprocessing and visualisation library. This 
experimental library contains additional classes to store data from the module.
The data types are built from the pandas DataFrame and are designed to superseed 
them for CHAPSim_post to enable some additional high level functionality to the
use and the other modules to allow data to be automatically reshaped when the 
__getitem__ method is used
"""

import itertools
import warnings
import numpy as np
import numbers
import copy

from abc import abstractmethod
from .io import hdfHandler
from ._api import rcParams
from .utils import find_stack_level
from .index import structIndexer

# from .coords import AxisData

# _HANDLE_NP_FUNCS = {}

# def implements(numpy_function):
#     """Register an __array_function__ implementation for MyArray objects."""
#     def decorator(func):
#         _HANDLE_NP_FUNCS[numpy_function] = func
#         return func
#     return decorator


class _StructMath(np.lib.mixins.NDArrayOperatorsMixin):

    _HANDLE_TYPES = (np.ndarray, numbers.Number)
    _ALLOWED_METHODS = ('__call__')
    _NOT_ALLOWED_UFUNCS = ()
    _NOT_ALLOWED_KWARGS = ('axis', 'out', 'axes')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):

        if method not in self._ALLOWED_METHODS:
            return NotImplemented

        if ufunc in self._NOT_ALLOWED_UFUNCS:
            return NotImplemented

        func = getattr(ufunc, method)
        nargs = len(inputs)

        if nargs == 1:
            return self._process_unary(func, inputs[0], **kwargs)

        else:
            return self._process_binary(func, *inputs, **kwargs)

    # def __array_function__(self, func, types, args, kwargs):
    #     if func not in _HANDLE_NP_FUNCS:
    #         return NotImplemented
    #     # Note: this allows subclasses that don't override
    #     # __array_function__ to handle MyArray objects

    #     return _HANDLE_NP_FUNCS[func](*args, **kwargs)

    @abstractmethod
    def _process_unary(self, func, input, **kwargs):
        pass

    @abstractmethod
    def _process_binary(self, func, *inputs, **kwargs):
        pass


class datastruct(_StructMath):

    def __init__(self, *args, from_hdf=False, **kwargs):

        from_array = False
        from_dict = False
        from_dstruct = False

        if isinstance(args[0], np.ndarray):
            from_array = True

        elif isinstance(args[0], dict):
            from_dict = True

        elif isinstance(args[0], datastruct):
            from_dstruct = True

        elif not from_hdf:
            msg = (f"No valid initialisation method for the {datastruct.__name__}"
                   " type has been found from arguments")
            raise ValueError(msg)

        if from_array:
            self._array_ini(*args, **kwargs)

        elif from_dict:
            self._dict_ini(*args, **kwargs)

        elif from_dstruct:
            self._dstruct_ini(*args, **kwargs)

        elif from_hdf:
            self._file_extract(*args, **kwargs)

    @classmethod
    def from_hdf(cls, *args, **kwargs):
        return cls(*args, from_hdf=True, **kwargs)

    def from_internal(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    @classmethod
    def from_concat(cls, struct_list):
        if not all(isinstance(struct, cls) for struct in struct_list):
            msg = f"All list elements must be isinstancces of the class {cls.__name__}"
            raise TypeError

        dstruct = struct_list[0].copy()
        for struct in struct_list[1:]:
            dstruct.concat(struct)

        return dstruct

    def _check_array_index(self, array, index):
        if len(array) != len(index):
            msg = "The length of the input array must match the index"
            raise ValueError(msg)

    def _array_ini(self, array, index=None, dtype=None, copy=False):
        if index is None:
            index = range(len(array))

        self._indexer = structIndexer(index)
        self._check_array_index(array, self._indexer)

        self._data = list(self._get_data(array, copy, dtype))

    def _dict_ini(self, dict_data, dtype=None, copy=False):
        if not all([isinstance(val, np.ndarray) for val in dict_data.values()]):
            msg = "The type of the values of the dictionary must be a numpy array"
            raise TypeError(msg)

        index = list(dict_data.keys())
        self._data = [self._get_data(data, copy, dtype)
                      for data in dict_data.values()]

        self._indexer = structIndexer(index)

    def _dstruct_ini(self, dstruct, copy=False):
        return self._dict_ini(dstruct.to_dict(), copy=copy)

    def _get_dtype(self, data, dtype=None):
        if dtype is None:
            if issubclass(data.dtype.type, np.floating):
                dtype = rcParams['dtype']
            else:
                dtype = data.dtype.type
        else:
            if isinstance(dtype, str):
                dtype = np.dtype(str)

            super_dtype = data.dtype.type.mro()[1]
            if not issubclass(dtype, super_dtype):
                msg = (f"Cannot set the dtype to {dtype.__name__}. "
                       f"Must be subclass of {super_dtype.__name__}")
                raise TypeError(msg)
        return dtype

    def _get_data(self, data, copy=False, dtype=None):
        dtype = self._get_dtype(data, dtype=dtype)

        return data.astype(dtype, copy=copy)

    def _file_extract(self, filename, key=None):
        hdf_obj = hdfHandler(filename, mode='r', key=key)

        hdf_obj.check_type_id(self.__class__)
        data_array = list(self._get_data(hdf_obj['data'][:], copy=True))
        index_array = hdf_obj.contruct_index_from_hdfkey('index')
        shapes_array = hdf_obj['shapes'][:]

        fill_func = self._fill_check(hdf_obj)
        if 'ndims' in hdf_obj.keys():
            ndims = hdf_obj['ndims'][:]
        elif 'ndims' in hdf_obj.attrs.keys():
            ndims = np.array(hdf_obj.attrs['ndims'])
        else:
            ndims = np.full(shapes_array.shape[0], shapes_array.shape[-1])

        self._data = [None for _ in range(len(data_array))]
        if self._require_fill(shapes_array):
            for i, (data, shape) in enumerate(zip(data_array, shapes_array)):
                self._data[i] = data[~fill_func(
                    data)].reshape(shape[:ndims[i]])
        else:
            for i, (data, shape) in enumerate(zip(data_array, shapes_array)):
                self._data[i] = data.reshape(shape[:ndims[i]])

        self._indexer = structIndexer(index_array)
        return hdf_obj

    def _require_fill(self, shapes):
        for shape in shapes[1:]:
            if not np.array_equal(shapes[0], shape):
                return True

        return False

    def _fill_check(self, h5_obj):
        if not 'fill_val' in h5_obj.attrs:
            return np.isnan

        fill = h5_obj.attrs['fill_val']

        def isneginf(x: np.ndarray):
            if issubclass(x.dtype.type, np.complexfloating):
                return np.logical_or(np.isneginf(x.real), np.isneginf(x.imag))
            else:
                return np.isneginf(x)

        if np.isnan(fill):
            return np.isnan
        elif np.isneginf(fill):
            return isneginf
        else:
            raise Exception("Need to add more fill values. Big sad.")

    def to_hdf(self, filepath, mode='a', key=None):
        hdf_obj = hdfHandler(filepath, mode=mode, key=key)

        fill_val = self._get_fill_val()
        hdf_shapes = self._construct_shapes_array()
        hdf_array = self._construct_data_array(fill_val, hdf_shapes)
        hdf_indices = self._indexer.to_array(string=True)

        hdf_obj.attrs['fill_val'] = fill_val
        hdf_obj.create_dataset('ndims',
                               data=np.array([a.ndim for a in self._data]))
        hdf_obj.create_dataset('data', data=hdf_array)
        hdf_obj.create_dataset('shapes', data=hdf_shapes)
        hdf_obj.create_dataset('index', data=hdf_indices)

        return hdf_obj

    def to_dict(self):
        return dict(self)

    def __mathandle__(self):
        out = dict()
        if self._is_multidim():
            for i in self.inner_index:
                full_array = np.stack([self[t, i]
                                      for t in self.outer_index], axis=0)
                out[i] = full_array

            out['times'] = np.array(self.outer_index)
        else:
            for i in self.inner_index:
                out[i] = self[i]
        return out

    def _get_fill_val(self):
        def isneginf(x: np.ndarray):
            if issubclass(x.dtype.type, np.complexfloating):
                return np.logical_or(np.isneginf(x.real), np.isneginf(x.imag))
            else:
                return np.isneginf(x)

        if not any(np.isnan(d).any() for d in self._data):
            return np.nan
        elif not any(isneginf(d).any() for d in self._data):
            return np.NINF
        else:
            raise Exception("Need to add more fill values. Big sad.")

    def _construct_shapes_array(self):
        shapes = [x.shape for _, x in self]
        max_dim = max([len(x) for x in shapes])
        for i, shape in enumerate(shapes):
            shape_mismatch = max_dim - len(shape)
            if shape_mismatch != 0:
                assert shape_mismatch >= 0
                extra_shape = [1]*shape_mismatch
                shapes[i] = [*shape, *extra_shape]
        return np.array(shapes)

    def _construct_data_array(self, fill_val, shapes):

        if self._require_fill(shapes):
            array = [x.flatten() for _, x in self]
            sizes = [x.size for x in array]
            max_size = max(sizes)
            outer_size = len(array)

            dtype = self._get_dtype(array[0])
            data_array = np.full((outer_size, max_size), fill_val, dtype=dtype)
            for i, arr in enumerate(array):
                data_array[i, :arr.size] = arr
        else:
            shape = (len(self.index), self._data[0].size)
            data_array = self.values.reshape(shape)

        return data_array

    def _is_multidim(self):
        return self._indexer.is_MultiIndex

    def equals(self, other_datastruct):
        if not isinstance(other_datastruct, datastruct):
            msg = "other_datastruct must be of type datastruct"
            raise TypeError(msg)

        for key, val1 in self:
            if key not in other_datastruct.index:
                return False
            if not np.allclose(val1, other_datastruct[key]):
                return False
        return True

    @property
    def index(self):
        return self._indexer

    @property
    def outer_index(self):
        return self._indexer.get_outer_index()

    @property
    def inner_index(self):
        return self._indexer.get_inner_index()

    @property
    def values(self):
        shape_list = [x.shape for x in self._data]
        if not all(x == shape_list[0] for x in shape_list):
            msg = "To use this function all the arrays in the datastruct must be the same shape"
            raise AttributeError(msg)

        return np.stack(self._data, axis=0)

    def __str__(self):
        return self._data.__str__()

    def __repr__(self):
        return self._data.__str__()

    def get_key(self, key):
        loc = self._indexer.get_loc(key)
        return self._data[loc]

    def __getitem__(self, key):
        if self._indexer.is_listkey(key):
            return self._getitem_process_list(key)
        elif self._indexer.is_multikey(key):
            return self._getitem_process_multikey(key)
        else:
            return self._getitem_process_singlekey(key)

    def _getitem_process_multikey(self, key):
        if not self._indexer.is_MultiIndex:
            msg = "A multidimensional index passed but a single dimensional datastruct"
            raise KeyError(msg)

        try:
            return self.get_key(key)
        except KeyError:
            err = KeyError((f"The key provided ({key}) to the datastruct is"
                            " not present and cannot be corrected internally."))
            warn = UserWarning((f"The outer index provided is incorrect ({key[0]})"
                                f" that is present (there is only one value present in the"
                                f" datastruct ({self._indexer.get_outer_index()[0]})"))

            inner_key = self.check_inner(key[1], err)
            outer_key = self.check_outer(key[0], err, warn)

            return self.get_key((outer_key, inner_key))

    def _getitem_process_singlekey(self, key):

        err_msg = (f"The key provided ({key}) to ""the datastruct is "
                   "not present and cannot be corrected internally.")
        err = KeyError(err_msg)

        try:
            return self.get_key(key)
        except KeyError:
            if self.index.is_MultiIndex:
                outer_key = self.check_outer(None, err)
                return self.get_key((outer_key, key))
            else:
                raise KeyError(err_msg) from None

    def _getitem_process_list(self, key):
        if self._is_multidim() and isinstance(key, list):
            key = (self.outer_index, key)

        key_list = self._indexer._getitem_process_list(key)

        struct_dict = {k: self[k] for k in key_list}
        return self.from_internal(struct_dict)

    def check_outer(self, key, err, warn=None):
        return self._indexer.check_outer(key, err, warn=warn)

    def check_inner(self, key, err):
        return self._indexer.check_inner(key, err)

    def __setitem__(self, key, value):
        if not isinstance(value, np.ndarray):
            msg = f"The input array must be an instance of {np.ndarray.__name__}"
            raise TypeError(msg)

        self.set_value(key, value)

    def set_value(self, key, value):
        if key in self.index:

            loc = self._indexer.get_loc(key)
            self._data[loc] = value
        else:
            self.create_new_item(key, value)

    def create_new_item(self, key, value):
        if key in self.index:
            msg = "Cannot create new key, the key is already present"
            raise KeyError(msg)

        if self._indexer.is_multikey(key) and not self._indexer.is_MultiIndex:
            msg = "Multi-key given, but the indexer is not a multiindex"
            raise TypeError(msg)

        if not self._indexer.is_multikey(key) and self._indexer.is_MultiIndex:
            msg = "single-key given, but the indexer is a multiindex"
            raise TypeError(msg)

        self._data.append(value)
        self._indexer.append(key)

    def __delitem__(self, key):

        if self._is_multidim() and not self._indexer.is_multikey(key):
            self.delete_inner_key(key)
            return

        if not key in self.index:
            raise KeyError(f"Key {key} not present in "
                           f"{self.__class__.__name__}")

        loc = self._indexer.get_loc(key)

        self._data.pop(loc)
        self._indexer.remove(key)

    def delete_inner_key(self, inner_key):
        if not inner_key in self.inner_index:
            msg = "Only inner keys in the inner index can be removed"
            raise KeyError(msg)
        outer_indices = self.outer_index
        for outer in outer_indices:
            del self[outer, inner_key]

    def delete_outer_key(self, outer_key):
        if not outer_key in self.outer_index:
            msg = "Only outer keys in the outer index can be removed"
            raise KeyError(msg)
        inner_indices = self.inner_index
        for inner in inner_indices:
            del self[outer_key, inner]

    def __iter__(self):
        for key, val in zip(self._indexer, self._data):
            yield (key, val)

    def iterref(self):
        return zip(self._indexer, self._data)

    def concat(self, arr_or_data):
        msg = f"`arr_or_data' must be of type {self.__class__.__name__} or an iterable of it"

        if isinstance(arr_or_data, self.__class__):
            if any(index in self.index for index in arr_or_data.index):
                e_msg = ("Key exists in current datastruct cannot concatenate")
                raise ValueError(e_msg)

            self._indexer.extend(arr_or_data._indexer)

            self._data.extend(arr_or_data._data)

        elif all([isinstance(arr, self.__class) for arr in arr_or_data]):
            for arr in arr_or_data:
                self.concat(arr)
        else:
            raise TypeError(msg)

    def append(self, arr, key=None, axis=0):
        if isinstance(arr, np.ndarray):
            msg = f"If the type of arr is {np.ndarray.__name__}, key must be provided"
            # raise TypeError(msg)
            if key is None:
                raise ValueError(msg)
            if len(self[key].shape) == 1:
                self[key] = np.stack([self[key], arr], axis=axis)
            else:
                self[key] = np.append(self[key], [arr], axis=axis)
        elif isinstance(arr, datastruct):
            if key is None:
                key = self.index
            if hasattr(key, "__iter__") and not isinstance(key, str):
                for k in key:
                    self.append(arr[k], key=k, axis=axis)
            else:
                self.append(arr[key], key=key, axis=axis)
        else:
            msg = f"Type of arr must be either {np.ndarray.__name__} or {datastruct.__name__}"
            raise TypeError(msg)

    def _process_unary(self, func, input, **kwargs):
        new_data = {key: func(val, **kwargs) for key, val in input}
        return datastruct(new_data)

    def _process_binary(self, func, *inputs, **kwargs):

        if isinstance(inputs[0], datastruct):
            this = inputs[0]
            other = inputs[1]

        else:
            this = inputs[1]
            other = inputs[0]

        if isinstance(other, datastruct):
            if not this.index == other.index:
                msg = "This can only be used if the indices in both datastructs are the same"
                raise ValueError(msg)

            new_data = {}
            for key, val in self:
                new_data[key] = func(val, other[key], **kwargs)

        else:
            try:
                new_data = {key: func(val, other, **kwargs)
                            for key, val in self}

            except TypeError:
                msg = (f"Cannot use operation {func.__name__} datastruct by "
                       f"object of type {type(other)}")
                raise TypeError(msg) from None

        return datastruct(new_data)

    # def _arith_binary_op(self,other_obj,func):
    #     if isinstance(other_obj,datastruct):
    #         if not self.index==other_obj.index:
    #             msg = "This can only be used if the indices in both datastructs are the same"
    #             raise ValueError(msg)
    #         new_data = {}
    #         for key, val in self:
    #             new_data[key] = func(val,other_obj[key])

    #     else:
    #         try:
    #             new_data = {key :func(val,other_obj) for key, val in self}
    #         except TypeError:
    #             msg = (f"Cannot use operation {func.__name__} datastruct by "
    #                     f"object of type {type(other_obj)}")
    #             raise TypeError(msg) from None

    #     return datastruct(new_data)

    # def __add__(self,other_obj):
    #     return self._arith_binary_op(other_obj,operator.add)

    # def __radd__(self,other_obj):
    #     return self.__add__(other_obj)

    # def __sub__(self,other_obj):
    #     return self._arith_binary_op(other_obj,operator.sub)

    # def __rsub__(self,other_obj):
    #     self_neg = operator.neg(self)
    #     return operator.add(self_neg,other_obj)

    # def __mul__(self,other_obj):
    #     return self._arith_binary_op(other_obj,operator.mul)

    # def __rmul__(self,other_obj):
    #     return self.__mul__(other_obj)

    # def __truediv__(self,other_obj):
    #     return self._arith_binary_op(other_obj,operator.truediv)

    # def _arith_unary_op(self,func):

    #     new_data = {key :func(val) for key, val in self}
    #     return datastruct(new_data)

    # def __abs__(self):
    #     return self._arith_unary_op(operator.abs)

    # def __neg__(self):
    #     return self._arith_unary_op(operator.neg)

    def __eq__(self, other_datastruct):
        return self.equals(other_datastruct)

    def __ne__(self, other_datastruct):
        return not self.equals(other_datastruct)

    def copy(self):
        cls = self.__class__
        return cls(copy.deepcopy(self.to_dict()))

    def __deepcopy__(self, memo):
        return self.copy()

    def __contains__(self, key):
        return key in self._indexer

    def symmetrify(self, dim=None):

        slicer = slice(None, None, None)
        indexer = [slicer for _ in range(self.values.ndim-1)]
        if dim is not None:
            indexer[dim] = slice(None, None, -1)

        data = {}
        for index, vals in self:

            comp = index[1] if self._is_multidim() else index
            count = comp.count('v') + comp.count('y')
            data[index] = vals.copy()[tuple(indexer)]*(-1)**count

        return datastruct(data)


class metastruct():
    def __init__(self, *args, from_hdf=False, **kwargs):

        from_list = False
        from_dict = False
        if isinstance(args[0], list):
            from_list = True
        elif isinstance(args[0], dict):
            from_dict = True
        elif not from_hdf:
            msg = (f"{self.__class__.__name__} can be instantiated by list,"
                   " dictionary or the class method from_hdf")
            raise TypeError(msg)

        if from_list:
            self._list_extract(*args, **kwargs)
        elif from_dict:
            self._dict_extract(*args, **kwargs)
        elif from_hdf:
            self._file_extract(*args, **kwargs)

    def _conversion(self, old_key, *replacement_keys):
        """
        Converts the old style metadata to the new style metadata
        """
        if old_key not in self._meta.keys():
            return

        if not isinstance(self._meta[old_key], list):
            item = [self._meta[old_key]]
        else:
            item = self._meta[old_key]
        for i, key in enumerate(replacement_keys):
            self._meta[key] = item[i]

        del self._meta[old_key]

    def _update_keys(self):
        update_dict = {
            'icase': ['iCase'],
            'thermlflg': ['iThermoDynamics'],
            'HX_tg_io': ['HX_tg', 'HX_io'],
            'NCL1_tg_io': ['NCL1_tg', 'NCL1_io'],
            'REINI_TIME': ['ReIni', 'TLgRe'],
            'FLDRVTP': ['iFlowDriven'],
            'CF': ['Cf_Given'],
            'accel_start_end': ['temp_start_end'],
            'HEATWALLBC': ['iThermalWallType'],
            'WHEAT0': ['thermalWallBC_Dim'],
            'RSTflg_tg_io': ['iIniField_tg', 'iIniField_io'],
            'RSTtim_tg_io': ['TimeReStart_tg', 'TimeReStart_io'],
            'RST_type_flg': ['iIniFieldType', 'iIniFieldTime'],
            'CFL': ['CFLGV'],
            'visthemflg': ['iVisScheme'],
            'Weightedpressure': ['iWeightedPre'],
            'DTSAVE1': ['dtSave1'],
            'TSTAV1': ['tRunAve1', 'tRunAve_Reset'],
            'ITPIN': ['iterMonitor'],
            'MGRID_JINI': ['MGRID', 'JINI'],
            'pprocessonly': ['iPostProcess'],
            'ppinst': ['iPPInst'],
            'ppspectra': ['iPPSpectra'],
            'ppdim': ['iPPDimension'],
            'ppinstnsz': ['pp_instn_sz'],
            'grad': ['accel_grad']
        }

        if 'NCL1_tg_io' in self._meta.keys() and 'iDomain' not in self._meta.keys():
            if self._meta['NCL1_tg_io'][1] < 2:
                self._meta['iDomain'] = 1
            else:
                self._meta['iDomain'] = 3

        if 'iCHT' not in self._meta.keys():
            self._meta['iCHT'] = 0

        if 'BCY12' not in self._meta.keys():
            self._meta['BCY12'] = [1, 1]

        if 'loc_start_end' in self._meta.keys():
            loc_list = [self._meta['loc_start_end'][0]*self._meta['HX_tg_io'][1],
                        self._meta['loc_start_end'][1]*self._meta['HX_tg_io'][1]]
            self._meta['location_start_end'] = loc_list
            del self._meta['loc_start_end']

        if 'DT' in self._meta.keys():
            if isinstance(self._meta['DT'], list):
                update_dict['DT'] = ['DT', 'DtMin']

        for key, val in update_dict.items():
            self._conversion(key, *val)

    def _list_extract(self, list_vals, index=None):
        if index is None:
            index = list(range(len(list_vals)))

        if len(index) != len(list_vals):
            msg = "The length of the index must be the same as list_vals"
            raise ValueError(msg)

        self._meta = {i: val for i, val in zip(index, list_vals)}
        self._update_keys()

    def _dict_extract(self, dictionary):
        self._meta = dictionary
        self._update_keys()

    def to_hdf(self, filename, key=None, mode='a'):
        hdf_obj = hdfHandler(filename, mode=mode, key=key)
        hdf_obj.set_type_id(self.__class__)
        str_items = hdf_obj.create_group('meta_str')

        for k, val in self._meta.items():

            if not hasattr(val, "__iter__") and not isinstance(val, str):
                hdf_obj.create_dataset(k, data=np.array([val]))
            else:
                if isinstance(val, str):
                    str_items.attrs[key] = val.encode('utf-8')
                else:
                    hdf_obj.create_dataset(k, data=np.array(val))

    @classmethod
    def from_hdf(cls, *args, **kwargs):
        return cls(*args, from_hdf=True, **kwargs)

    def _file_extract(self, filename, *args, key=None, **kwargs):
        hdf_obj = hdfHandler(filename, mode='r', key=key)

        index = list(key for key in hdf_obj.keys() if key != 'meta_str')
        list_vals = []
        for k in index:
            val = list(hdf_obj[k])
            if len(val) == 1:
                val = val[0]
            list_vals.append(val)

        index.extend(hdf_obj.attrs.keys())

        str_items = hdf_obj['meta_str'] if 'meta_str' in hdf_obj.keys(
        ) else hdf_obj
        for k in str_items.attrs.keys():
            list_vals.append(str_items.attrs[k])

        self._meta = {i: val for i, val in zip(index, list_vals)}

        self._update_keys()

    def __mathandle__(self):
        return self._meta

    @property
    def index(self):
        return self._meta.keys()

    def __getitem__(self, key):
        if key not in self._meta.keys():
            msg = "key not found in metastruct"
            raise KeyError(msg)
        return self._meta[key]

    def __setitem__(self, key, value):
        warn_msg = "item in metastruct being manually overriden, this may be undesireable"
        warnings.warn(warn_msg, stacklevel=find_stack_level())

        self._meta[key] = value

    def copy(self):
        cls = self.__class__
        index = list(self._meta.keys()).copy()
        values = list(self._meta.values()).copy()
        return cls(values, index=index)

    def __deepcopy__(self, memo):
        return self.copy()
