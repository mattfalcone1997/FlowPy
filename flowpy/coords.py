
from .core import *
import numpy as np
from .utils import check_list_vals
from .gradient import Grad_calc
from ._style import *

from pyvista import StructuredGrid

class GeomHandler(StyleHandler):
    def __init__(self,flowTYPE,style_params=None):
        
        self._check_flow_type(flowTYPE)
        
        self._geomTYPE = flowTYPE
        self.Grad_calc = Grad_calc

        super().__init__(flowTYPE,style_params=style_params)

    @staticmethod
    def _check_flow_type(geom_type):
        if geom_type not in [PIPE,CHANNEL,BLAYER]:
            msg = "geometry type not valid"
            raise ValueError(msg)
        
    @property
    def is_polar(self):
        return self._geomTYPE == PIPE
    
    @property
    def is_channel(self):
        return self._geomTYPE == CHANNEL

    def __str__(self):
        if self.is_channel:
            name = "channel"
        elif self.is_polar:
            name = "pipe"
        else:
            name = "boundary layer"
            
        return f"{self.__class__.__name__} instance with %s flow"%name
    
    def to_hdf_attr(self,h5_obj):
        h5_obj.attrs["GeomType"] = self._geomTYPE
        
    def __mathandle__(self):
        return {"GeomType" : self._geomTYPE}
    
    @classmethod
    def from_hdf_attr(cls,h5_obj):
        return cls(h5_obj.attrs['GeomType'])
    
    def __repr__(self):
        return self.__str__()

    def __deepcopy__(self,memo):
        return self.__class__(self._geomTYPE,dict(self._style_params))
        
    
class AxisData:
    _domain_handler_class = GeomHandler
    def __init__(self, *args,from_file=False, **kwargs):
        if from_file:
            self._hdf_extract(*args,**kwargs)
        else:
            self._coordstruct_extract(*args,**kwargs)
            

        self._check_integrity()
    
    def _check_integrity(self):
        if self.coord_staggered is None:
            return

        if self.coord_staggered.index != self.coord_centered.index:
            msg = "Indices of coordstructs must be the same"
            raise ValueError(msg)

        for x in self.coord_centered.index:
            size = self.coord_centered[x].size
            if self.coord_staggered[x].size != size + 1:
                msg = ("The shape of the staggered data if given must be"
                        " one greater than centered in each dimension")
                raise ValueError(msg)

            msg = "The staggered and centered coordinates must be interleaved"
            for i in range(size):
                if self.coord_centered[x][i] < self.coord_staggered[x][i]:
                    raise ValueError(msg)
                if self.coord_centered[x][i] > self.coord_staggered[x][i+1]:
                    raise ValueError(msg)
                
    @classmethod
    def from_hdf(cls,*args,**kwargs):
        return cls(*args,from_file=True,**kwargs)
    
    def _hdf_extract(self,filename,key=None):
        if key is None:
            key = self.__class__.__name__
        

        hdf_obj = hdfHandler(filename,mode='r',key=key)
        hdf_obj.check_type_id(self.__class__)

        self._domain_handler = self._domain_handler_class.from_hdf_attr(hdf_obj)

        self.coord_centered = coordstruct.from_hdf(filename,key=key+"/coord_centered")
        if 'coord_staggered' in hdf_obj.keys():
            self.coord_staggered = coordstruct.from_hdf(filename,key=key+"/coord_staggered")
        else:
            self.coord_staggered = None
                    
    def _coordstruct_extract(self,Domain,coord,coord_nd):
        self._domain_handler = Domain
        self.coord_staggered = coord_nd
        self.coord_centered = coord
        
    def to_hdf(self,filename,mode,key=None):
        if key is None:
            key = self.__class__.__name__

        self.coord_centered.to_hdf(filename,key=key+"/coord_centered",mode=mode)
        
        if self.contains_staggered:
            self.coord_staggered.to_hdf(filename,key=key+"/coord_staggered",mode=mode)

        hdf_obj = hdfHandler(filename,mode='r',key=key)
        self._domain_handler.to_hdf_attr(hdf_obj)
    
    def __mathandle__(self):
        out = dict()
        out['axis_data'] = self._domain_handler.__mathandle__()
        out['CoordDF'] = self.coord_centered.__mathandle__()
        if self.contains_staggered:
            out['Coord_ND_DF'] = self.coord_staggered.__mathandle__()

        return out
    
    def create_vtkStructuredGrid(self,staggered = True):
        if staggered:
            if not self.contains_staggered:
                msg = "The staggered data cannot be None if this options is set"
                raise ValueError(msg)

            x_coords = self.coord_staggered['x']
            y_coords = self.coord_staggered['y']
            z_coords = self.coord_staggered['z']
        else:
            x_coords = self.coord_centered['x']
            y_coords = self.coord_centered['y']
            z_coords = self.coord_centered['z']

        Y,X,Z = np.meshgrid(y_coords,x_coords,z_coords)

        grid = StructuredGrid(X,Z,Y)
        return grid
    
    @property
    def contains_staggered(self):
        return not self.coord_staggered is None
    
    @property
    def staggered(self):
        return self.coord_staggered

    @property
    def centered(self):
        return self.coord_centered
    
    def copy(self):
        return copy.deepcopy(self)
    
    
    def __eq__(self,other_obj):
        if not isinstance(other_obj,AxisData):
            msg = "This operation can only be done on other objects of this type"
            raise TypeError(msg)

        if self._domain_handler.is_polar != other_obj._domain_handler.is_polar:
            return False

        if self.coord_centered != other_obj.coord_centered:
            return False

        if self.coord_staggered != other_obj.coord_staggered:
            return False

        return True

    def __ne__(self,other_obj):
        return not self.__eq__(other_obj)
    
class coordstruct(datastruct):
    
    def set_domain_handler(self,GeomHandler):
        self._domain_handler = GeomHandler

    @property
    def DomainHandler(self):
        if hasattr(self,"_domain_handler"):
            return self._domain_handler
        else: 
            return None

    def Translate(self,**args):
        for key in args:
            if key in self.index:
                self[key] += args[key]
            else:
                raise KeyError("Invalid direction specfied "
                               "(%s). Directions %s "%(key,self.index))
                
    def Rescale(self,val):
        for key in self.index:
            self[key] /= val 
            
    def _get_subdomain_lims(self,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        if xmin is None:
            xmin = np.amin(self['x'])
        if xmax is None:
            xmax = np.amax(self['x'])
        if ymin is None:
            ymin = np.amin(self['y'])
        if ymax is None:
            ymax = np.amax(self['y'])
        if zmin is None:
            zmin = np.amin(self['z'])
        if zmax is None:
            zmax = np.amax(self['z'])
            
        xmin_index, xmax_index = (self.index_calc('x',xmin)[0],
                                    self.index_calc('x',xmax)[0])
        ymin_index, ymax_index = (self.index_calc('y',ymin)[0],
                                    self.index_calc('y',ymax)[0])
        zmin_index, zmax_index = (self.index_calc('z',zmin)[0],
                                    self.index_calc('z',zmax)[0])
        return xmin_index,xmax_index,ymin_index,ymax_index,zmin_index,zmax_index

    def create_subdomain(self,xmin=None,xmax=None,ymin=None,ymax=None,zmin=None,zmax=None):
        (xmin_index,xmax_index,
        ymin_index,ymax_index,
        zmin_index,zmax_index) = self._get_subdomain_lims(xmin,xmax,ymin,ymax,zmin,zmax)

        xcoords = self['x'][xmin_index:xmax_index]
        ycoords = self['y'][ymin_index:ymax_index]
        zcoords = self['z'][zmin_index:zmax_index]

        return self.__class__({'x':xcoords, 'y':ycoords,'z':zcoords})

    def index_calc(self,comp,coord_list):
        coords = self[comp]
        coord_list = check_list_vals(coord_list)
        
        min_coord = np.amin(coords)
        max_coord = np.amax(coords)

        index_list=[]
        for coord in coord_list:
            
            if min_coord == max_coord:
                index_list = [0]   
            elif coord < min_coord or coord > max_coord:
                end_threshold = abs(coords[-1] - coords[-2])
                start_threshold = abs(coords[1] - coords[0])
                
                if abs(max_coord - coord) < end_threshold:
                    index_list.append(coords.size-1)
                elif abs(min_coord - coord) < start_threshold:
                    index_list.append(0)
                else:
                    msg = "Value in coord_list out of bounds: "\
                        + "%s coordinate given %g. Coordinate range [%g,%g]" % (comp,coord,min_coord,max_coord)
                    raise IndexError(msg) from None
            else:
                min_array = np.abs(coords - coord)
                index_list.append(np.argmin(min_array))
                
                

        return index_list
    
    def get_true_coords(self,axis,coord_list):
        indices = self.index_calc(axis, coord_list)
        return self[axis][indices]

    def check_plane(self,plane):
        if plane not in ['xy','zy','xz']:
            plane = [ x[::-1] for x in plane ]
            if plane not in ['xy','zy','xz']:
                msg = "The contour slice must be either %s"%['xy','yz','xz']
                raise KeyError(msg)
        slice_set = set(plane)
        coord_set = set(list('xyz'))
        coord = "".join(coord_set.difference(slice_set))
        return plane, coord
            
    def check_line(self,line):
        if line not in self.index:
            msg = f"The line must be in {self.index}"
            raise KeyError(msg)

        return line
    
    def __mathandle__(self):
        return super().__mathandle__()

    def _process_binary(self, func, *inputs, **kwargs):
        dstruct = super()._process_binary(func, *inputs, **kwargs)
        return self.__class__(dstruct)

    def _process_unary(self, func, input, **kwargs):
        dstruct = super()._process_unary(func, input, **kwargs)
        return self.__class__(dstruct)