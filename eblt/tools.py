from __future__ import annotations

import datetime
import enum
import functools
import html
import importlib
import inspect
import logging
import pathlib
import string
import subprocess
import sys
import textwrap
import traceback
import typing
import uuid
from numbers import Number
from typing import Any, Dict, Generator, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import prettytable
import pydantic
import pydantic_settings

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







