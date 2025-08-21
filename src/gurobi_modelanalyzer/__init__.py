__version__ = "v2.1.0"

from .common import _config

from .results_analyzer import (
    kappa_explain,
    angle_explain,
    matrix_bitmap,
    converttofractions,
)

from .solcheck import SolCheck

set_env = _config.set_env
