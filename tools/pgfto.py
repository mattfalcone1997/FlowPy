
import subprocess
from shutil import which, copy

import os
from tempfile import TemporaryDirectory, NamedTemporaryFile

#------------------------------------------------

class ProcessError(RuntimeError):
    def __init__(self,out: subprocess.CompletedProcess):
        self._out = out

    def __str__(self) -> str:
        message = (f"Errorcode {self._out.returncode}\n"
                        "\tstderr returned:\n"
                        f"\t\t{self._out.stderr}")
        return message

class latexError(ProcessError):
    pass

class dvipsError(ProcessError):
    pass

class eps2epsError(ProcessError):
    pass

class PgfTo:
    def __init__(self,input_fn,
                 preamble=None,
                 latex_cmd='pdflatex',
                 doc_options=None,
                 extra_packages: dict=None,
                 dependent_files=None,
                 mpl_rasterized=False):

        if not os.path.isfile(input_fn):
            raise FileNotFoundError(f'{input_fn} not found')
        self._name = input_fn
        
        self._latex_cmd = latex_cmd

        self._raster_files = []
        if mpl_rasterized:
            base, fn = os.path.split(input_fn)

            root = os.path.splitext(fn)[0]
            raster_root = root + '-img'
            self._raster_files = [os.path.join(base,f) for f in os.listdir(base)\
                             if raster_root in f]

        self._doc_options = [] if doc_options is None else list(doc_options)

        if os.path.splitext(self._name)[-1] == 'pgf':
            self._extra_packages = {'pgf':None}
        elif os.path.splitext(self._name)[-1] == 'tex':
            self._doc_options.append('preview')

        self._extra_packages = {}
        if extra_packages is not None:
            for k, v in extra_packages.items():
                msg = ("Package options must be a "
                        "tuple or list of str")
                    
                if not isinstance(v,(tuple,list)):
                    raise TypeError(msg)
                
                if not all(isinstance(o,str) for o in v):
                    raise TypeError(msg)
            
                self._extra_packages[k] = v

        self._preamble = None
        if preamble is not None:
            if not os.path.isfile(preamble):
                raise FileNotFoundError(f'{preamble} not found')
            
            self._preamble = preamble
        
        if dependent_files is None:
            dependent_files = []

        elif not isinstance(dependent_files,list):
            raise TypeError("Dependent files must be of type list")

        for file in dependent_files:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"{file} does not exist")

        self._dependent_files = dependent_files

    def _create_temporary(self):
        cwd = os.getcwd()
        temp = TemporaryDirectory()

        copy(os.path.join(cwd,self._name),
             temp.name)
        
        for fn in self._raster_files:
            copy(os.path.join(cwd,fn),
                 temp.name)

        copy(os.path.join(cwd,self._preamble),
             temp.name)
        
        for fn in self._dependent_files:
            copy(os.path.join(cwd,fn),
                 temp.name)
            
        return cwd, temp



    def _build_document(self, cwd, temp):
        lines = []

        lines.append("\\documentclass[%s]{standalone}\n"%",".join(self._doc_options))
        if self._preamble is not None:
            lines.append('\\input{%s}\n'%self._preamble)

        for k, v in self._extra_packages.items():
            if v is None:
                lines.append('\\usepackage{%s}\n'%k)
            else:
                lines.append('\\usepackage[%s]{%s}\n'%(','.join(v),k))

        lines.append("\\begin{document}\n")
        lines.append("\t\\input{%s}\n"%os.path.basename(self._name))
        lines.append("\\end{document}\n")


        fn = NamedTemporaryFile(mode='w',
                                suffix='.tex',
                                dir=temp.name,
                                delete=False)
        
        fn.writelines(lines)

        fn.close()

        return fn.name



    def _run_latex(self,cwd,fn: str,dvi=False):

        # t_prefix = os.path.splitext(t.name)[0]

        dvi_option = '-output-format=dvi' if dvi else ""
        cmd = [self._latex_cmd,
                "-interaction=nonstopmode",
                dvi_option,
                 fn]

        out = subprocess.run(cmd,capture_output=True)

        t_prefix = os.path.splitext(fn)[0]
        prefix = os.path.splitext(self._name)[0]
        log_file = os.path.join(cwd,os.path.basename(prefix+'.log'))

        if out.returncode != 0:
            copy(t_prefix+'.log', log_file)
            raise latexError(out)
        else:
            if os.path.isfile(log_file):
                os.remove(log_file)

        if dvi:
            return t_prefix+'.dvi'
        else:
            return t_prefix+'.pdf'

    def _run_dvips(self,src,dst=None):

        if not which('dvips'):
            raise RuntimeError("dvips not found")
        
        if dst is None:
            dst = os.path.splitext(src)[0] + '.eps'
        cmds = ['dvips','-E',src,'-o',dst]
        out = subprocess.run(cmds,capture_output=True)

        if out.returncode != 0:
            dvipsError(out)

        return dst
    
    def _run_eps2eps(self,src,dst=None):
        if not which('eps2eps'):
            raise RuntimeError("eps2eps not found")

        if dst is None:
            dst = NamedTemporaryFile(dir=os.path.split(src)[0],
                                     suffix='.eps',
                                     delete=False)
        cmds = ['eps2eps',src,dst.name]
        out = subprocess.run(cmds,capture_output=True)
        if out.returncode != 0:
            raise eps2epsError(out)

        return dst.name
    
    def to_eps(self,output_fn=None):

        if output_fn is None:
            output_fn = os.path.splitext(self._name)[0] + '.eps'
        output_fn = os.path.abspath(output_fn)

        cwd, temp = self._create_temporary()
        
        fn = self._build_document(cwd, temp)
        os.chdir(temp.name)

        dvi_fn = self._run_latex(cwd,fn,dvi=True)

        eps_fn = self._run_dvips(dvi_fn)

        eps_fn2 = self._run_eps2eps(eps_fn)

        copy(eps_fn2,output_fn)
        os.chdir(cwd)

        temp.cleanup()

    def to_pdf(self,output_fn=None):
        if output_fn is None:
            output_fn = os.path.splitext(self._name)[0] + '.eps'
        output_fn = os.path.abspath(output_fn)

        cwd, temp = self._create_temporary()

        fn = self._build_document(cwd, temp)
        os.chdir(temp.name)

        pdf_fn = self._run_latex(cwd,fn,dvi=False)

        copy(pdf_fn,output_fn)
        os.chdir(cwd)
        temp.cleanup()
