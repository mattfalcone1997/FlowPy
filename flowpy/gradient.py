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
from numba import njit, prange, jit
from math import prod


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

def gradient_numba_compact(flow_array: np.ndarray, coord_array: np.ndarray,axis=-1):
    if axis < 0:
        axis = flow_array.ndim+axis
    

    dx = (coord_array[-1] - coord_array[0])/(len(coord_array)-1)
    x = np.linspace(coord_array[0],coord_array[-1],len(coord_array))

    slices = get_slices(flow_array.shape,axis,flow_array.ndim)

    rhs = _get_rhs_compact(flow_array,dx,slices)

    source = list(range(flow_array.ndim))
    dst = source.copy(); dst.remove(axis);dst.append(axis)

    rhs_swap = np.moveaxis(rhs,source,dst)
    shape = (prod(rhs_swap.shape[:-1]),rhs_swap.shape[-1])

    matrix_coeffs = _get_matrix_coeffs(coord_array)
    grad = _compute_tdma(rhs_swap.reshape(shape),*matrix_coeffs)
    grad_swap = grad.reshape(rhs_swap.shape)
    
    return np.moveaxis(grad_swap,dst,source)

@njit(parallel=True)
def _compute_tdma(flow_swap: np.ndarray,a: np.ndarray, b: np.ndarray, c: np.ndarray):
    grad = np.zeros_like(flow_swap)
    n = flow_swap.shape[1]
    w= np.zeros(n,a.dtype)
    b1= np.zeros(n,a.dtype)
    
    b1[0] = b[0]
    for i in range(1,n):
        w[i] = a[i]/b1[i-1]
        b1[i] = b[i]-w[i]*c[i-1]

    for j in prange(flow_swap.shape[0]):
        d= np.zeros(n, a.dtype)
        d[0] = flow_swap[j,0]
        for i in range(1,n):
            d[i] = flow_swap[j,i]-w[i]*d[i-1]

        grad[j,n-1] = d[n-1]/b1[n-1]
        for i in range(n-2,-1,-1):
            grad[j,i] = (d[i]-c[i]*grad[j,i+1])/b1[i]

    return grad

def _get_matrix_coeffs(coords: np.ndarray):
    coeffs = np.zeros((3,coords.size))
    coeffs[1,0] = 1; coeffs[2,0] = 2
    coeffs[0,1] = 0.25; coeffs[1,1] = 1; coeffs[2,1] = 0.25
    coeffs[0,2:-2] = 1/3; coeffs[1,2:-2] = 1; coeffs[2,2:-2] = 1/3
    coeffs[0,-2] = 0.25; coeffs[1,-2] = 1; coeffs[2,-2] = 0.25
    coeffs[0,-1] = 2; coeffs[1,-1] = 1
    return coeffs

def get_slices(shape,axis,ndim):
    slicer = [slice(None)]*ndim
    edge1_slice = slicer.copy()
    edge2_slice = slicer.copy()
    edge3_slice = slicer.copy()

    edge1_slice[axis]=slice(0,1)
    edge2_slice[axis]=slice(1,2)
    edge3_slice[axis] = slice(2,3)

    edge1_slice = tuple(edge1_slice)
    edge2_slice = tuple(edge2_slice)
    edge3_slice = tuple(edge3_slice)

    edgen1_slice = slicer.copy()
    edgen2_slice = slicer.copy()
    edgen3_slice = slicer.copy()

    edgen1_slice[axis]= slice(shape[axis] -1,shape[axis])
    edgen2_slice[axis]= slice(shape[axis] -2,shape[axis]-1)
    edgen3_slice[axis] =slice(shape[axis] -3,shape[axis]-2)

    edgen1_slice = tuple(edgen1_slice)
    edgen2_slice = tuple(edgen2_slice)
    edgen3_slice = tuple(edgen3_slice)

    middle1_slice = slicer.copy()
    middle2_slice = slicer.copy()
    middle_slice = slicer.copy()
    middle3_slice = slicer.copy()
    middle4_slice = slicer.copy()
    

    middle1_slice[axis] = slice(0,shape[axis] -4)
    middle2_slice[axis] = slice(1,shape[axis] -3)
    middle_slice[axis] = slice(2,shape[axis] -2)
    middle3_slice[axis] = slice(3,shape[axis] -1)
    middle4_slice[axis] = slice(4,shape[axis] )

    middle1_slice = tuple(middle1_slice)
    middle2_slice = tuple(middle2_slice)
    middle_slice = tuple(middle_slice)
    middle3_slice = tuple(middle3_slice)
    middle4_slice = tuple(middle4_slice)

    slices = (edge1_slice,
              edge2_slice,
              edge3_slice,
              middle1_slice,
              middle2_slice,
              middle_slice,
              middle3_slice,
              middle4_slice,
              edgen1_slice,
              edgen2_slice,
              edgen3_slice)
    return slices

@njit(parallel=True)
def _get_rhs_compact(flow_array: np.ndarray,dx,slices):
    a= 7/9/dx
    b = 1/36/dx
    rhs = np.zeros_like(flow_array)
    rhs[slices[0]] = (-2.5*flow_array[slices[0]] +2.*flow_array[slices[1]] \
                        +0.5*flow_array[slices[2]])/dx
    rhs[slices[1]] = (0.75/dx)*(flow_array[slices[2]] -flow_array[slices[0]])

    # boundary at n, n-1



    rhs[slices[8]] = (2.5*flow_array[slices[8]] -2.*flow_array[slices[9]] \
                        -0.5*flow_array[slices[10]])/dx
    rhs[slices[9]] = (0.75/dx)*(flow_array[slices[8]] -flow_array[slices[10]])
    

    # middle
    rhs[slices[5]] = a*(flow_array[slices[6]] - flow_array[slices[4]]) \
                      + b*(flow_array[slices[7]] - flow_array[slices[3]])

    return rhs

    
    

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