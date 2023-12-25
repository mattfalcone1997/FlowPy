# """
# Copied and modified from matplotlib v3.7.2
# """
import shutil
import pathlib
import os
import functools
import weakref
from subprocess import run, PIPE, Popen
from tempfile import TemporaryDirectory
import matplotlib.font_manager as fm
from matplotlib import (rcParams,
                        cbook)
from matplotlib._version import __version__
from matplotlib.rcsetup import validate_string
from matplotlib.backends.backend_pgf import (FigureCanvasPgf,
                                             FigureManagerPgf,
                                             RendererPgf,
                                             _create_pdf_info_dict,
                                             _metadata_to_str,
                                             _log,
                                             _writeln,
                                             _tex_escape,
                                             mpl_pt_to_in,
                                             LatexError)
from matplotlib.backends.backend_mixed import MixedModeRenderer
if __version__ < "3.6":
    from matplotlib.backends.backend_pgf import get_preamble as _get_preamble
else:
    from matplotlib.backends.backend_pgf import _get_preamble

def _validate_preamble(s: str):
    if s is None:
        return

    if not os.path.isfile(s):
        raise FileNotFoundError(f"{s} not found")

    if not os.path.splitext(s)[-1] == '.tex':
        raise ValueError("File extension must be '.tex'")
    
    return s

def _validate_tex_args(s: list):
    if isinstance(s, list):
        if not all(isinstance(a, str) for a in s):
            raise TypeError("Must be list of str of str")
        return s
    elif isinstance(s,str):
        return [s]
    else:
        raise TypeError("Invalid type")
    
def get_preamble():
    preamble = _get_preamble()

    if rcParams['pgf.preamble_file'] is not None:
        with open(rcParams['pgf.preamble_file'], 'r') as f:
            preamble_file = "".join(f.readlines())
        preamble = '\n'.join([preamble,preamble_file])
    return preamble

def update_rcParams():
    if "pgf.document_class" not in rcParams:
        _new_validators = {"pgf.document_class": validate_string,
                        "pgf.preamble_file": _validate_preamble,
                        "pgf.tex_args": _validate_tex_args}
        rcParams.validate.update(_new_validators)

        rcParams["pgf.document_class"] = 'article'
        rcParams["pgf.preamble_file"] = None
        rcParams["pgf.tex_args"] = ""


class LatexManager:
    """
    The LatexManager opens an instance of the LaTeX application for
    determining the metrics of text elements. The LaTeX environment can be
    modified by setting fonts and/or a custom preamble in `.rcParams`.
    """

    @staticmethod
    def _build_latex_header():        
        latex_header = [
            r"\documentclass{%s}"%rcParams.get('pgf.document_class', 'article'),
            # Include TeX program name as a comment for cache invalidation.
            # TeX does not allow this to be the first line.
            rf"% !TeX program = {rcParams['pgf.texsystem']}",
            # Test whether \includegraphics supports interpolate option.
            r"\usepackage{graphicx}",
            get_preamble(),
            r"\begin{document}",
            r"\typeout{pgf_backend_query_start}",
        ]
        return "\n".join(latex_header)

    @classmethod
    def _get_cached_or_new(cls):
        """
        Return the previous LatexManager if the header and tex system did not
        change, or a new instance otherwise.
        """
        return cls._get_cached_or_new_impl(cls._build_latex_header())

    @classmethod
    @functools.lru_cache(1)
    def _get_cached_or_new_impl(cls, header):  # Helper for _get_cached_or_new.
        return cls()

    def _stdin_writeln(self, s):
        if self.latex is None:
            self._setup_latex_process()
        self.latex.stdin.write(s)
        self.latex.stdin.write("\n")
        self.latex.stdin.flush()

    def _expect(self, s):
        s = list(s)
        chars = []
        while True:
            c = self.latex.stdout.read(1)
            chars.append(c)
            if chars[-len(s):] == s:
                break
            if not c:
                self.latex.kill()
                self.latex = None
                raise LatexError("LaTeX process halted", "".join(chars))
        return "".join(chars)

    def _expect_prompt(self):
        return self._expect("\n*")

    def __init__(self):
        # create a tmp directory for running latex, register it for deletion
        self._tmpdir = TemporaryDirectory()
        self.tmpdir = self._tmpdir.name
        self._finalize_tmpdir = weakref.finalize(self, self._tmpdir.cleanup)

        doc_class = rcParams.get('pgf.document_class', 'article')
        if os.path.isfile(doc_class+'.cls'):
                shutil.copy(doc_class+'.cls', self._tmpdir.name)
        # test the LaTeX setup to ensure a clean startup of the subprocess
        self._setup_latex_process(expect_reply=False)
        stdout, stderr = self.latex.communicate("\n\\makeatletter\\@@end\n")
        if self.latex.returncode != 0:
            raise LatexError(
                f"LaTeX errored (probably missing font or error in preamble) "
                f"while processing the following input:\n"
                f"{self._build_latex_header()}",
                stdout)
        self.latex = None  # Will be set up on first use.
        # Per-instance cache.
        self._get_box_metrics = functools.lru_cache(self._get_box_metrics)

    def _setup_latex_process(self, *, expect_reply=True):
        # Open LaTeX process for real work; register it for deletion.  On
        # Windows, we must ensure that the subprocess has quit before being
        # able to delete the tmpdir in which it runs; in order to do so, we
        # must first `kill()` it, and then `communicate()` with it.
        try:
            self.latex = Popen(
                [rcParams["pgf.texsystem"], "-halt-on-error"],
                stdin=PIPE, stdout=PIPE,
                encoding="utf-8", cwd=self.tmpdir)
        except FileNotFoundError as err:
            raise RuntimeError(
                f"{rcParams['pgf.texsystem']!r} not found; install it or change "
                f"rcParams['pgf.texsystem'] to an available TeX implementation"
            ) from err
        except OSError as err:
            raise RuntimeError(
                f"Error starting {rcParams['pgf.texsystem']!r}") from err

        def finalize_latex(latex):
            latex.kill()
            latex.communicate()

        self._finalize_latex = weakref.finalize(
            self, finalize_latex, self.latex)
        # write header with 'pgf_backend_query_start' token
        self._stdin_writeln(self._build_latex_header())
        if expect_reply:  # read until 'pgf_backend_query_start' token appears
            self._expect("*pgf_backend_query_start")
            self._expect_prompt()

    def get_width_height_descent(self, text, prop):
        """
        Get the width, total height, and descent (in TeX points) for a text
        typeset by the current LaTeX environment.
        """
        return self._get_box_metrics(_escape_and_apply_props(text, prop))

    def _get_box_metrics(self, tex):
        """
        Get the width, total height and descent (in TeX points) for a TeX
        command's output in the current LaTeX environment.
        """
        # This method gets wrapped in __init__ for per-instance caching.
        self._stdin_writeln(  # Send textbox to TeX & request metrics typeout.
            # \sbox doesn't handle catcode assignments inside its argument,
            # so repeat the assignment of the catcode of "^" and "%" outside.
            r"{\catcode`\^=\active\catcode`\%%=\active\sbox0{%s}"
            r"\typeout{\the\wd0,\the\ht0,\the\dp0}}"
            % tex)
        try:
            answer = self._expect_prompt()
        except LatexError as err:
            # Here and below, use '{}' instead of {!r} to avoid doubling all
            # backslashes.
            raise ValueError("Error measuring {}\nLaTeX Output:\n{}"
                             .format(tex, err.latex_output)) from err
        try:
            # Parse metrics from the answer string.  Last line is prompt, and
            # next-to-last-line is blank line from \typeout.
            width, height, offset = answer.splitlines()[-3].split(",")
        except Exception as err:
            raise ValueError("Error measuring {}\nLaTeX Output:\n{}"
                             .format(tex, answer)) from err
        w, h, o = float(width[:-2]), float(height[:-2]), float(offset[:-2])
        # The height returned from LaTeX goes from base to top;
        # the height Matplotlib expects goes from bottom to top.
        return w, h + o, o

class FigureCanvas(FigureCanvasPgf):
    filetypes = {"pgf": "LaTeX PGF picture",
                 "pdf": "LaTeX compiled PGF picture",
                 "png": "Portable Network Graphics",
                 "ps": "PostScript",
                 "eps": "Encapsulated Postscript"}
    
    def get_renderer(self):
        return Renderer(self.figure, None)

    def _print_pgf_to_fh(self, fh, *, bbox_inches_restore=None, **kwargs):

        header_text = """%% Creator: Matplotlib, PGF backend
%%
%% To include the figure in your LaTeX document, write
%%   \\input{<filename>.pgf}
%%
%% Make sure the required packages are loaded in your preamble
%%   \\usepackage{pgf}
%%
%% Also ensure that all the required font packages are loaded; for instance,
%% the lmodern package is sometimes necessary when using math font.
%%   \\usepackage{lmodern}
%%
%% Figures using additional raster images can only be included by \\input if
%% they are in the same directory as the main LaTeX file. For loading figures
%% from other directories you can use the `import` package
%%   \\usepackage{import}
%%
%% and then include the figures with
%%   \\import{<path to file>}{<filename>.pgf}
%%
"""

        # append the preamble used by the backend as a comment for debugging
        header_info_preamble = ["%% Matplotlib used the following preamble"]
        for line in _get_preamble().splitlines():
            header_info_preamble.append("%%   " + line)
        header_info_preamble.append("%%")
        header_info_preamble = "\n".join(header_info_preamble)

        # get figure size in inch
        w, h = self.figure.get_figwidth(), self.figure.get_figheight()
        dpi = self.figure.dpi

        # create pgfpicture environment and write the pgf code
        fh.write(header_text)
        fh.write(header_info_preamble)
        fh.write("\n")
        _writeln(fh, r"\begingroup")
        _writeln(fh, r"\makeatletter")
        _writeln(fh, r"\begin{pgfpicture}")
        _writeln(fh,
                 r"\pgfpathrectangle{\pgfpointorigin}{\pgfqpoint{%fin}{%fin}}"
                 % (w, h))
        _writeln(fh, r"\pgfusepath{use as bounding box, clip}")
        renderer = MixedModeRenderer(self.figure, w, h, dpi,
                                     Renderer(self.figure, fh),
                                     bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)

        # end the pgfpicture environment
        _writeln(fh, r"\end{pgfpicture}")
        _writeln(fh, r"\makeatother")
        _writeln(fh, r"\endgroup")
    def _print_latex_output(self, fmt, fname_or_fh, *, metadata=None, **kwargs):
        """Use LaTeX to compile a pgf generated figure to pdf."""
        w, h = self.figure.get_size_inches()

        info_dict = _create_pdf_info_dict('pgf', metadata or {})
        pdfinfo = ','.join(
            _metadata_to_str(k, v) for k, v in info_dict.items())

        doc_class = rcParams.get('pgf.document_class', 'article')

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

        # print figure to pgf and compile it with latex
        with TemporaryDirectory() as tmpdir:
            tmppath = pathlib.Path(tmpdir)
            if os.path.isfile(doc_class+'.cls'):
                shutil.copy(doc_class+'.cls', tmppath)

            self.print_pgf(tmppath / "figure.pgf", **kwargs)
            (tmppath / "figure.tex").write_text(
                "\n".join([
                    r"\documentclass[12pt]{%s}" % doc_class,
                    r"\usepackage[pdfinfo={%s}]{hyperref}" % pdfinfo,
                    r"\usepackage[papersize={%fin,%fin}, margin=0in]{geometry}"
                    % (w, h),
                    r"\pagenumbering{gobble}",
                    r"\usepackage{pgf}",
                    get_preamble(),
                    r"\begin{document}",
                    r"\centering",
                    r"\input{figure.pgf}",
                    r"\end{document}",
                ]), encoding="utf-8")
            
            texcommand = rcParams["pgf.texsystem"]

            cbook._check_and_log_subprocess(
                [texcommand, "-interaction=nonstopmode", "-halt-on-error",
                  *rcParams["pgf.tex_args"], "figure.tex"], _log, cwd=tmpdir)
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

def _escape_and_apply_props(s, prop):
    """
    Generate a TeX string that renders string *s* with font properties *prop*,
    also applying any required escapes to *s*.
    """
    commands = []

    families = {"serif": r"\rmfamily", "sans": r"\sffamily",
                "sans-serif": r"\sffamily", "monospace": r"\ttfamily"}
    family = prop.get_family()[0]
    if family in families:
        commands.append(families[family])
    elif any(font.name == family for font in fm.fontManager.ttflist):
        commands.append(
            r"\ifdefined\pdftexversion\else\setmainfont{%s}\rmfamily\fi" % family)
    else:
        _log.warning("Ignoring unknown font: %s", family)

    size = prop.get_size_in_points()
    commands.append(r"\fontsize{%f}{%f}" % (size, size * 1.2))

    styles = {"normal": r"", "italic": r"\itshape", "oblique": r"\slshape"}
    commands.append(styles[prop.get_style()])

    boldstyles = ["semibold", "demibold", "demi", "bold", "heavy",
                  "extra bold", "black"]
    if prop.get_weight() in boldstyles:
        commands.append(r"\bfseries")

    commands.append(r"\selectfont")
    return (
        "{"
        + "".join(commands)
        + r"\catcode`\^=\active\def^{\ifmmode\sp\else\^{}\fi}"
        # It should normally be enough to set the catcode of % to 12 ("normal
        # character"); this works on TeXLive 2021 but not on 2018, so we just
        # make it active too.
        + r"\catcode`\%=\active\def%{\%}"
        + _tex_escape(s)
        + "}"
    )


class Renderer(RendererPgf):
    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        # docstring inherited

        # prepare string for tex
        s = _escape_and_apply_props(s, prop)

        _writeln(self.fh, r"\begin{pgfscope}")
        self._print_pgf_clip(gc)

        alpha = gc.get_alpha()
        if alpha != 1.0:
            _writeln(self.fh, r"\pgfsetfillopacity{%f}" % alpha)
            _writeln(self.fh, r"\pgfsetstrokeopacity{%f}" % alpha)
        rgb = tuple(gc.get_rgb())[:3]
        _writeln(self.fh, r"\definecolor{textcolor}{rgb}{%f,%f,%f}" % rgb)
        _writeln(self.fh, r"\pgfsetstrokecolor{textcolor}")
        _writeln(self.fh, r"\pgfsetfillcolor{textcolor}")
        s = r"\color{textcolor}" + s

        dpi = self.figure.dpi
        text_args = []
        if mtext and (
                (angle == 0 or
                 mtext.get_rotation_mode() == "anchor") and
                mtext.get_verticalalignment() != "center_baseline"):
            # if text anchoring can be supported, get the original coordinates
            # and add alignment information
            pos = mtext.get_unitless_position()
            x, y = mtext.get_transform().transform(pos)
            halign = {"left": "left", "right": "right", "center": ""}
            valign = {"top": "top", "bottom": "bottom",
                      "baseline": "base", "center": ""}
            text_args.extend([
                f"x={x/dpi:f}in",
                f"y={y/dpi:f}in",
                halign[mtext.get_horizontalalignment()],
                valign[mtext.get_verticalalignment()],
            ])
        else:
            # if not, use the text layout provided by Matplotlib.
            text_args.append(f"x={x/dpi:f}in, y={y/dpi:f}in, left, base")

        if angle != 0:
            text_args.append("rotate=%f" % angle)

        _writeln(self.fh, r"\pgftext[%s]{%s}" % (",".join(text_args), s))
        _writeln(self.fh, r"\end{pgfscope}")

    def get_text_width_height_descent(self, s, prop, ismath):
        # docstring inherited
        # get text metrics in units of latex pt, convert to display units
        w, h, d = (LatexManager._get_cached_or_new()
                   .get_width_height_descent(s, prop))
        # TODO: this should be latex_pt_to_in instead of mpl_pt_to_in
        # but having a little bit more space around the text looks better,
        # plus the bounding box reported by LaTeX is VERY narrow
        f = mpl_pt_to_in * self.dpi
        return w * f, h * f, d * f
    
FigureManager = FigureManagerPgf
