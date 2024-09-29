from __future__ import annotations


import logging

import subprocess

import traceback

from typing import  Union
from .types import AnyPath, NDArray
import enum
import numpy as np
import prettytable
import pydantic_settings
import functools
import sys
import importlib
import os
import json
from hashlib import blake2b

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

try:
    from types import UnionType
except ImportError:
    # Python < 3.10
    union_types = {Union}
else:
    union_types = {UnionType, Union}


logger = logging.getLogger(__name__)


class DisplayOptions(
    pydantic_settings.BaseSettings,
    env_prefix="LUME_",
    case_sensitive=False,
):
    """
    jupyter_render_mode : One of {"html", "markdown", "genesis", "repr"}
        Defaults to "repr".
        Environment variable: LUME_JUPYTER_RENDER_MODE.
    console_render_mode : One of {"markdown", "genesis", "repr"}
        Defaults to "repr".
        Environment variable: LUME_CONSOLE_RENDER_MODE.
    include_description : bool, default=True
        Include descriptions in table representations.
        Environment variable: LUME_INCLUDE_DESCRIPTION.
    ascii_table_type : int, default=prettytable.MARKDOWN
        Default to a PrettyTable markdown ASCII table.
        Environment variable: LUME_ASCII_TABLE_TYPE.
    filter_tab_completion : bool, default=True
        Filter out unimportant details (pydantic methods and such) from
        Genesis4 classes.
        Environment variable: LUME_FILTER_TAB_COMPLETION.
    verbose : int, default=0
        At level 0, hide Genesis4 output during `run()` by default.
        At level 1, show Genesis4 output during `run()` by default.
        Equivalent to configuring the default setting of `Genesis4.verbose` to
        `True`.
        Environment variable: LUME_VERBOSE.
    """

    jupyter_render_mode: Literal["html", "markdown", "genesis", "repr"] = "repr"
    console_render_mode: Literal["markdown", "genesis", "repr"] = "repr"
    include_description: bool = True
    ascii_table_type: int = prettytable.MARKDOWN
    verbose: int = 0
    filter_tab_completion: bool = True


global_display_options = DisplayOptions()

class NpEncoder(json.JSONEncoder):
    """
    See: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class OutputMode(enum.Enum):
    """Jupyter Notebook output support."""

    unknown = "unknown"
    plain = "plain"
    html = "html"

@functools.cache
def get_output_mode() -> OutputMode:
    """
    Get the output mode for lume-genesis objects.

    This works by way of interacting with IPython display and seeing what
    choice it makes regarding reprs.

    Returns
    -------
    OutputMode
        The detected output mode.
    """
    if "IPython" not in sys.modules or "IPython.display" not in sys.modules:
        return OutputMode.plain

    from IPython.display import display

    class ReprCheck:
        mode: OutputMode = OutputMode.unknown

        def _repr_html_(self) -> str:
            self.mode = OutputMode.html
            return "<!-- lume-genesis detected Jupyter and will use HTML for rendering. -->"

        def __repr__(self) -> str:
            self.mode = OutputMode.plain
            return ""

    check = ReprCheck()
    display(check)
    return check.mode


def is_jupyter() -> bool:
    """Is Jupyter detected?"""
    return get_output_mode() == OutputMode.html

def safe_loadtxt(filepath: AnyPath, **kwargs) -> NDArray:
    """
    Similar to np.loadtxt, but handles old-style exponents d -> e
    """
    s = open(filepath).readlines()
    s = list(map(lambda x: x.lower().replace('d', 'e'), s))
    return np.loadtxt(s, **kwargs)


def execute(cmd, cwd=None):
    """
    Constantly print Subprocess output while process is running
    from: https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running

    # Example usage:
        for path in execute(["locate", "a"]):
        print(path, end="")

    Useful in Jupyter notebook

    """
    popen = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, universal_newlines=True, cwd=cwd
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


# Alternative execute
def execute2(cmd, timeout=None, cwd=None):
    """
    Execute with time limit (timeout) in seconds, catching run errors.
    """
    output = {"error": True, "log": ""}
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=timeout,
            cwd=cwd,
        )
        output["log"] = p.stdout
        output["error"] = False
        output["why_error"] = ""
    except subprocess.TimeoutExpired as ex:
        stdout = ex.stdout or b""
        output["log"] = "\n".join((stdout.decode(), f"{ex.__class__.__name__}: {ex}"))
        output["why_error"] = "timeout"
    except subprocess.CalledProcessError as ex:
        stdout = ex.stdout or b""
        output["log"] = "\n".join((stdout.decode(), f"{ex.__class__.__name__}: {ex}"))
        output["why_error"] = "error"
    except Exception as ex:
        stack = traceback.print_exc()
        output["log"] = f"Unknown run error: {ex.__class__.__name__}: {ex}\n{stack}"
        output["why_error"] = "unknown"
    return output

def import_by_name(clsname: str) -> type:
    """
    Import the given class or function by name.

    Parameters
    ----------
    clsname : str
        The module path to find the class e.g.
        ``"pcdsdevices.device_types.IPM"``

    Returns
    -------
    type
    """
    module, cls = clsname.rsplit(".", 1)
    if module not in sys.modules:
        importlib.import_module(module)

    mod = sys.modules[module]
    try:
        return getattr(mod, cls)
    except AttributeError:
        raise ImportError(f"Unable to import {clsname!r} from module {module!r}")

def full_path(path):
    """
    Helper function to expand enviromental variables and return the absolute path
    """
    return os.path.abspath(os.path.expandvars(os.path.expanduser(path)))

def class_key_data(E):
     keyed_data = {key: value for key, value in E.__dict__.items() if not key.startswith('__') and not callable(key)}
     return keyed_data


def update_hash(keyed_data, h):
    """
    Creates a cryptographic fingerprint from keyed data.
    Used JSON dumps to form strings, and the blake2b algorithm to hash.

    """
    for key in sorted(keyed_data.keys()):
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
        
def fingerprint(keyed_data, digest_size=16):
    """
    Creates a cryptographic fingerprint from keyed data.
    Used JSON dumps to form strings, and the blake2b algorithm to hash.

    """
    h = blake2b(digest_size=16)
    for key in sorted(keyed_data.keys()):
        val = keyed_data[key]
        s = json.dumps(val, sort_keys=True, cls=NpEncoder).encode()
        h.update(s)
    return h.hexdigest()