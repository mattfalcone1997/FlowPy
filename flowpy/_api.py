
from functools import wraps
import numpy as np
from matplotlib import docstring

from .plotting import FlowAxes, FlowFigure
from matplotlib import RcParams
from matplotlib.rcsetup import (validate_bool)
import warnings

def _validate_dtype(s):
    if isinstance(s,str):
            s = np.dtype(s).type
    
    if not issubclass(s,np.floating):
        msg = f"dtype must be of type or str corresponding to numpy floating type"
        raise TypeError(msg)
    
    return s

class InvalidGradientMethod(ValueError): pass

def _validate_gradient(s):
    if not s in ['cython', 'numpy']:
        raise InvalidGradientMethod("Gradient can either be cython or numpy")
    return s

def _validate_cupy(s):
    s = validate_bool(s)
    try:
        if s:
            import cupy
            
            if cupy.is_available():
                return s
            else:
                warnings.warn("There are no CUDA GPUs available")
                return s
        else:
            return s
    except (ImportError, ModuleNotFoundError):
        warnings.warn('There has been a problem import CuPy'
                     ' check installation this has been deactivated')
        return False


_param_dict = {"dtype": np.dtype('f8'),
               "gradient_method":"cython",
               "relax_HDF_type_checks":False,
               "use_cupy" : False}

_validate_params = { "dtype" :_validate_dtype,
                     "gradient_method" : _validate_gradient,
                     "relax_HDF_type_checks" : validate_bool,
                     "use_cupy" : _validate_cupy}

def get_rcparams():
    params = RcParams()
    params.validate = _validate_params
    dict.update(params,_param_dict)
    return params
    
rcParams = get_rcparams()

handle = docstring.Substitution()

sub = handle
copy = docstring.copy

short_name={
    "ndarray": np.ndarray.__name__,
    "ax" : FlowAxes.__name__,
    "fig" : FlowFigure.__name__
}
handle.update(**short_name)

def inherit(method):

    @wraps(method)
    def func(self,*args,**kwargs):
        for parent in self.__class__.__mro__[1:]:
            source = getattr(parent,method.__name__,None)
            if source is not None:
                __doc__ = source.__doc__ ; break

        return method(self,*args,**kwargs)

    return func
                


# inherit = docInherit

def copy_fromattr(attr):
    """Copy a docstring from another source function (if present)."""
    def decorator(func):   
        @wraps(func)
        def from_attr(*args,**kwargs):
            self = args[0]
            attr_func = getattr(self,attr)
            __doc__ = attr_func.__doc__

            return func(*args,**kwargs)
        return from_attr
    return decorator
