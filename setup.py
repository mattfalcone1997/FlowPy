
if __name__ == "__main__":
    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import Extension
    import numpy
    import os
    from Cython.Build import cythonize
    
    #---------------------------------------------------------------
    """
    Building and install flowpy
    """
    def create_cython_ext(folder,**other_args):
        sources = [os.path.join(folder,file) for file in os.listdir(folder) \
                        if os.path.splitext(file)[-1] == '.pyx']
        names = [os.path.splitext(source)[0].replace('/','.')\
                    for source in sources]
        include_dirs = [numpy.get_include()]
        if 'include_dirs' in other_args:
            other_args['include_dirs'] += include_dirs
        else:
            other_args['include_dirs'] = include_dirs
        ext_list = []
        for name, source in zip(names,sources):
            ext_list.append(Extension(name=name,
                                      sources=[source],
                                      **other_args))
            
        return ext_list
        
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)

    cy_parallel = create_cython_ext("flowpy/_libs",
                                    extra_compile_args = ["-fopenmp","-O3"],
                                    extra_link_args = ["-fopenmp","-O3"])
    

    ext_list = cythonize(cy_parallel,
                         compiler_directives={'language_level':3})

    config = Configuration(package_name='flowpy',
                            description="Initial code for general structured grid fluid postprocessing tool",
                            package_path="flowpy",
                            ext_modules = ext_list)
        
    setup(**config.todict())