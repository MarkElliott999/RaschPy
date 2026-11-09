"""
Progress reporting for the slow methods: CML fits, bootstrap resamples, Monte Carlo p-values, model selection.

One line, updated in place on stderr (terminal and Jupyter): done / total, elapsed, estimated time left. The public
methods take progress= (via with_progress): None (default) shows the line only when the estimated run time exceeds
THRESHOLD seconds, from the time of the first units; True always; False never. A run estimated to take longer than
LONG seconds gives one UserWarning with the estimate and, where there is one, the faster alternative. Reporters
started inside another one (the CML fits of a bootstrap's refits) stay silent.
"""

import contextvars
import functools
import inspect
import sys
import time
import warnings

THRESHOLD = 10.0  # seconds: progress=None shows the line when the estimated run time exceeds this
LONG = 600.0  # seconds: one UserWarning when the estimated run time exceeds this
_REFRESH = 0.5  # seconds between redraws

_setting = contextvars.ContextVar("raschpy_progress", default=None)
_active = contextvars.ContextVar("raschpy_progress_active", default=False)

DOC = """
        progress : {None, True, False}, default None
            Progress line for the slow parts (CML fits, bootstrap resamples,
            Monte Carlo samples, model-selection fits): done / total,
            elapsed and estimated time left, updated in place. None shows it
            when the estimated run time exceeds 10 seconds; True always;
            False never. A run estimated to take over 10 minutes also gives
            one UserWarning with the estimate.
"""


class option:
    """Context manager: the progress setting for everything run inside it (None keeps the outer setting)."""

    def __init__(self, progress):
        if progress not in (None, True, False):
            raise ValueError(f"progress must be None, True or False, got {progress!r}.")
        self.progress, self.token = progress, None

    def __enter__(self):
        if self.progress is not None:
            self.token = _setting.set(self.progress)
        return self

    def __exit__(self, *exc):
        if self.token is not None:
            _setting.reset(self.token)
        return False


def with_progress(func):
    """Decorator: adds a keyword-only progress= argument (and its docstring entry) to a public method."""
    sig = inspect.signature(func)
    params = list(sig.parameters.values())
    if "progress" not in sig.parameters:
        new = inspect.Parameter("progress", inspect.Parameter.KEYWORD_ONLY, default=None)
        at = next((i for i, p in enumerate(params) if p.kind == p.VAR_KEYWORD), len(params))
        params.insert(at, new)

    @functools.wraps(func)
    def wrapper(*args, progress=None, **kwargs):
        with option(progress):
            return func(*args, **kwargs)

    wrapper.__signature__ = sig.replace(parameters=params)
    wrapper.__doc__ = _with_doc(func.__doc__ or "")
    return wrapper


def _with_doc(doc):
    """doc with the progress entry at the end of its Parameters section (else at the end)."""
    import re

    p = re.search(r"\n[ ]*Parameters\n[ ]*-+\n", doc)
    if p:
        nxt = re.search(r"\n[ ]*(Returns|Attributes set|Attributes|Notes|Raises|Examples|See Also)\n[ ]*-+\n", doc[p.end():])
        if nxt:
            at = p.end() + nxt.start()
            return doc[:at].rstrip() + "\n" + DOC.rstrip() + "\n" + doc[at:]
    return doc.rstrip() + "\n" + DOC


def _fmt(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else (f"{m}m {s:02d}s" if m else f"{s}s")


class Reporter:
    """
    Progress of one slow loop. total=None for an open-ended loop (an iterative fit), shown once it has run for
    THRESHOLD / 3 seconds under progress=None. Use as a context manager; update() after each unit (note: extra
    text, e.g. the fit's change). alternative: the faster option the long-run warning names.
    """

    def __init__(self, label, total=None, unit="", alternative=None, stacklevel=3):
        self.label, self.total, self.unit = label, total, unit
        self.alternative, self.stacklevel = alternative, stacklevel
        self.setting = _setting.get()
        self.nested = _active.get()
        self.enabled = not self.nested
        self.done, self.shown, self.warned, self.last = 0, False, False, 0.0
        self.t0, self.token, self.width = time.perf_counter(), None, 0

    def __enter__(self):
        if self.enabled:
            self.token = _active.set(True)
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def estimate(self):
        el = time.perf_counter() - self.t0
        if self.total and self.done:
            return el, el / self.done * self.total
        return el, None

    def update(self, n=1, note=""):
        if not self.enabled:
            return
        self.done += n
        el, est = self.estimate()
        if est is not None and est > LONG and not self.warned:
            self.warned = True
            extra = f" {self.alternative}" if self.alternative else ""
            warnings.warn(
                f"{self.label}: estimated run time about {_fmt(est)} ({self.total} {self.unit}).{extra}",
                UserWarning,
                stacklevel=self.stacklevel,
            )
        if self.setting is False:
            return
        show = self.setting is True or (
            est > THRESHOLD if est is not None else el > THRESHOLD / 3
        )
        now = time.perf_counter()
        if show and (now - self.last >= _REFRESH or (self.total and self.done >= self.total)):
            self.last = now
            self._draw(el, est, note)

    def _draw(self, el, est, note):
        if self.total:
            left = max(est - el, 0.0) if est is not None else 0.0
            text = f"{self.label}: {self.done}/{self.total} {self.unit} | elapsed {_fmt(el)} | about {_fmt(left)} left"
        else:
            text = f"{self.label}: {self.done} {self.unit} | elapsed {_fmt(el)}"
        if note:
            text += f" | {note}"
        pad = " " * max(self.width - len(text), 0)
        self.width = len(text)
        sys.stderr.write("\r" + text + pad)
        sys.stderr.flush()
        self.shown = True

    def close(self):
        if self.token is not None:
            _active.reset(self.token)
            self.token = None
        if self.shown:
            el, est = self.estimate()
            self._draw(el, est, "done")
            sys.stderr.write("\n")
            sys.stderr.flush()
            self.shown = False


def track(iterable, label, total=None, unit="", alternative=None, stacklevel=4):
    """A loop over iterable with a progress line (Reporter), updated after each item; closed when the loop ends,
    breaks or raises."""
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    rep = Reporter(label, total=total, unit=unit, alternative=alternative, stacklevel=stacklevel).__enter__()
    try:
        for x in iterable:
            yield x
            rep.update()
    finally:
        rep.close()
