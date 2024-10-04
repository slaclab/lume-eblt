from .tools import safe_loadtxt
import numpy as np
import os
from .types import AnyPath
from typing import Dict


def read_fieldmap_rfdata(Path: AnyPath, file_id: float) -> Dict:
    """
    Read BELT rfdata file, which should be simple four-column ASCII data
    """

    info = {}
    info['format'] = 'rfdata'
    info['filename'] = 'rfdata' + str(int(file_id))

    filepath = os.path.abspath(os.path.join(Path, info['filename']))
    info['filePath'] = filepath

    assert os.path.isfile(filepath), "Wakefield file " + filepath + "  not found"

    # Read data
    d = {}
    d['info'] = info
    d['data'] = safe_loadtxt(filepath)
    return d


def write_fieldmap_rfdata(filePath: AnyPath, fieldmap: Dict) -> None:
    """

    """
    np.savetxt(filePath, fieldmap['data'])
