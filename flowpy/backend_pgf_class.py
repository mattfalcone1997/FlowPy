# """
# Copied and modified from matplotlib v3.7.2
# """
import shutil
import pathlib
import os
import functools
from subprocess import run
from tempfile import TemporaryDirectory

from matplotlib import (rcParams,
                        cbook)
from matplotlib._version import __version__
from matplotlib.rcsetup import validate_string
from matplotlib.backends.backend_pgf import (FigureCanvasPgf,
                                             FigureManagerPgf,
                                             _create_pdf_info_dict,
                                             _metadata_to_str,
                                             _log,
                                             LatexError)
if __version__ < "3.6":
    from matplotlib.backends.backend_pgf import get_preamble as _get_preamble
else:
    from matplotlib.backends.backend_pgf import _get_preamble

FigureManager = FigureManagerPgf


def _validate_preamble(s: str):
    if s is None:
        return

    if not os.path.isfile(s):
        raise FileNotFoundError(f"{s} not found")

    if not os.path.splitext(s)[-1] == '.tex':
        raise ValueError("File extension must be '.tex'")


def update_rcParams():
    _new_validators = {"pgf.document_class": validate_string,
                       "pgf.preamble_file": _validate_preamble}
    rcParams.validate.update(_new_validators)
    rcParams["pgf.document_class"] = 'article'
    rcParams["pgf.preamble_file"] = None


class FigureCanvas(FigureCanvasPgf):
    filetypes = {"pgf": "LaTeX PGF picture",
                 "pdf": "LaTeX compiled PGF picture",
                 "png": "Portable Network Graphics",
                 "ps": "PostScript",
                 "eps": "Encapsulated Postscript"}

    def _print_pgf_to_fh(self, fh, *, bbox_inches_restore=None, **kwargs):
        return super()._print_pgf_to_fh(fh, bbox_inches_restore=bbox_inches_restore)

    def _print_latex_output(self, fmt, fname_or_fh, *, metadata=None, **kwargs):
        """Use LaTeX to compile a pgf generated figure to pdf."""
        w, h = self.figure.get_size_inches()

        info_dict = _create_pdf_info_dict('pgf', metadata or {})
        pdfinfo = ','.join(
            _metadata_to_str(k, v) for k, v in info_dict.items())

        doc_class = rcParams.get('pgf.default_class', 'article')

        if fmt in ['ps', 'eps']:
            if not shutil.which("pdftops"):
                raise RuntimeError(f"Format {self.filetypes[fmt]} requires "
                                   "requires pdftops to be installed")

        if not shutil.which('kpsewhich'):
            raise LatexError("kpsewhich not found")

        cmds = ['kpsewhich', doc_class + '.cls']
        out = run(cmds, capture_output=True)
        if not out.stdout:
            raise LatexError("Latex document class not found")

        if rcParams['pgf.preamble_file'] is not None:
            preamble_input = r"\input{%s}" % rcParams['pgf.preamble_file']
        else:
            preamble_input = ""

        # print figure to pgf and compile it with latex
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            if os.path.isfile(doc_class+'.cls'):
                shutil.copy(doc_class+'.cls', tmppath)
            if rcParams['pgf.preamble_file'] is not None:
                shutil.copy(rcParams['pgf.preamble_file'], tmppath)

            self.print_pgf(tmppath / "figure.pgf", **kwargs)
            (tmppath / "figure.tex").write_text(
                "\n".join([
                    r"\documentclass[12pt]{%s}" % doc_class,
                    r"\usepackage[pdfinfo={%s}]{hyperref}" % pdfinfo,
                    r"\usepackage[papersize={%fin,%fin}, margin=0in]{geometry}"
                    % (w, h),
                    r"\pagenumbering{gobble}",
                    r"\usepackage{pgf}",
                    _get_preamble(),
                    preamble_input,
                    r"\begin{document}",
                    r"\centering",
                    r"\input{figure.pgf}",
                    r"\end{document}",
                ]), encoding="utf-8")
            texcommand = rcParams["pgf.texsystem"]

            cbook._check_and_log_subprocess(
                [texcommand, "-interaction=nonstopmode", "-halt-on-error",
                 "figure.tex"], _log, cwd=tmpdir)
            if fmt == 'pdf':
                with (tmppath / "figure.pdf").open("rb") as orig, \
                        cbook.open_file_cm(fname_or_fh, "wb") as dest:
                    # copy file contents to target
                    shutil.copyfileobj(orig, dest)
            else:

                if fmt == 'ps':
                    command = ['pdftops', "figure.pdf", "figure.ps"]
                else:
                    command = ['pdftops', '-eps', "figure.pdf", "figure.eps"]

                cbook._check_and_log_subprocess(
                    command, _log, cwd=tmpdir)
                with (tmppath / command[-1]).open("rb") as orig, \
                        cbook.open_file_cm(fname_or_fh, "wb") as dest:
                    # copy file contents to target
                    shutil.copyfileobj(orig, dest)

    print_pdf = functools.partialmethod(_print_latex_output, 'pdf')
    print_ps = functools.partialmethod(_print_latex_output, 'ps')
    print_eps = functools.partialmethod(_print_latex_output, 'eps')
