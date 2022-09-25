"""
## gradient.py
Module for calculating gradient in CHAPSim_post.
Currently uses numpy.gradient as its core. It may contain a
 variety of methodologies in the future.
"""
import warnings

import numpy as np
from ._api import rcParams
from ._libs import gradient


__all__ = ["Grad_calc"] #,'Scalar_grad_io',"Vector_div_io",
            #"Scalar_laplacian_io","Scalar_laplacian_tg",
            # "totIntegrate_y",'cumIntegrate_y','Curl']

_cupy_avail = False

try:
    import cupy as cnpy
    _cupy_avail = cnpy.is_available()
except:
    pass
class Gradient():
    def __init__(self):
        self.__type = None
    
    def setup(self,coordDF):
        if self.__type is None or self.__type != rcParams['gradient_method']:
            self.__type = rcParams['gradient_method']
            attr_name = "setup_" + rcParams['gradient_method'] + "_method"
            try:
                func = getattr(self,attr_name)
            except AttributeError:
                msg = "The stated gradient calculation method is invalid"
                raise ValueError(msg) from None
            func = getattr(self,attr_name)
            func(coordDF)

    def setup_numpy_method(self,coordDF):
        self.grad_calc_method = self.grad_calc_numpy

    def setup_cython_method(self,coordDF):
        self.grad_calc_method = self.grad_calc_cy

    def setup_cupy_method(self,coordDF):
        self.grad_calc_method = self.grad_calc_cupy

    def _get_grad_dim(self,flow_array,comp):
        if flow_array.ndim == 3:
            dim = ord('z') - ord(comp)
        elif flow_array.ndim == 2:
            dim = ord('y') - ord(comp)
            if comp == 'z':
                msg = "gradients in the z direction can only be calculated on 3-d arrays"
                raise ValueError(msg)
        elif flow_array.ndim == 1:
            dim = 0
            if comp != 'y':
                msg = "For 1D flow arrays only y can be used"
                raise ValueError(msg)
        else:
            msg = "This method can only be used on one, two and three dimensional arrays"
            raise TypeError(msg)
        
        return dim
            
    def grad_calc_numpy(self,CoordDF,flow_array,comp):

        dim = self._get_grad_dim(flow_array,comp)
        coord_array = CoordDF[comp]
        
        if coord_array.size != flow_array.shape[dim]:
            msg = (f"The coordinate array size ({coord_array.size})"
                    f" and flow array size in dimension ({flow_array.shape[dim]})"
                    " does not match")
            raise ValueError(msg)
            
        return np.gradient(flow_array,coord_array,edge_order=2,axis=dim)
    
    def grad_calc_cy(self,CoordDF,flow_array,comp):
        
        dim = self._get_grad_dim(flow_array,comp)

        coord_array = CoordDF[comp]
        if coord_array.size != flow_array.shape[dim]:
            msg = (f"The coordinate array size ({coord_array.size})"
                    f" and flow array size in dimension ({flow_array.shape[dim]})"
                    " does not match")
            raise ValueError(msg)
        
        return gradient.gradient_calc(flow_array.astype('f8'),
                                      coord_array.astype('f8'),dim)
    
    def grad_calc_cupy(self,CoordDF,flow_array,comp):
        
        
        msg = "This method cannot be called if cupy is not available"
        if not _cupy_avail: raise RuntimeError(msg)
                
        dim = self._get_grad_dim(flow_array,comp)

        coord_array = CoordDF[comp]
        if coord_array.size != flow_array.shape[dim]:
            msg = (f"The coordinate array size ({coord_array.size})"
                    f" and flow array size in dimension ({flow_array.shape[dim]})"
                    " does not match")
            raise ValueError(msg)
            
        return cnpy.gradient(flow_array,coord_array,edge_order=2,axis=dim).get()
    
    def Grad_calc(self,coordDF,flow_array,comp):
        self.setup(coordDF)
        return self.grad_calc_method(coordDF,flow_array,comp)

Gradient_method = Gradient()

Grad_calc = Gradient_method.Grad_calc


def Scalar_grad(coordDF,flow_array):

    if flow_array.ndim == 2:
        grad_vector = np.zeros((2,flow_array.shape[0],flow_array.shape[1]))
    elif flow_array.ndim == 3:
        grad_vector = np.zeros((3,flow_array.shape[0],flow_array.shape[1],
                               flow_array.shape[2]))
    else:
        msg = "This function can only be used on 2 or 3 dimensional arrays"
        raise ValueError(msg)

    grad_vector[0] = Grad_calc(coordDF,flow_array,'x')
    grad_vector[1] = Grad_calc(coordDF,flow_array,'y')
    
    
    if flow_array.ndim == 3:
        grad_vector[2] = Grad_calc(coordDF,flow_array,'z')
        
    return grad_vector


def Vector_div(coordDF,vector_array):
    if vector_array.ndim not in (3,4):
        msg = "The number of dimension of the vector array must be 3 or 4"
        raise ValueError(msg)

    grad_vector = np.zeros_like(vector_array)
    grad_vector[0] = Grad_calc(coordDF,vector_array[0],'x')
    grad_vector[1] = Grad_calc(coordDF,vector_array[1],'y')
    
    
    if vector_array.ndim == 4:
        grad_vector[2] = Grad_calc(coordDF,vector_array[2],'z')
        
    
    div_scalar = np.sum(grad_vector,axis=0)
    
    return div_scalar

def Scalar_laplacian(coordDF,flow_array):
    grad_vector = Scalar_grad(coordDF,flow_array)
    lap_scalar = Vector_div(coordDF,grad_vector)
    return lap_scalar


def Curl(CoordDF,flow_array,polar=True):

    curl_array = np.zeros_like(flow_array)
    if flow_array.shape[0] == 3:
        if polar:
            r = CoordDF['y'][:,np.newaxis]
            r_inv = r**-1
        else:
            r = 1.
            r_inv = 1.

        curl_array[0] = r_inv*(Grad_calc(CoordDF,flow_array[2],'y') -\
                            Grad_calc(CoordDF,flow_array[1],'z'))
        curl_array[1] = r_inv*Grad_calc(CoordDF,flow_array[2],'x') -\
                            Grad_calc(CoordDF,flow_array[0],'z')
        curl_array[2] = Grad_calc(CoordDF,flow_array[1],'x') -\
                            Grad_calc(CoordDF,flow_array[0],'y')

        return curl_array
    else:
        raise ValueError("Must be 3D big sad")    