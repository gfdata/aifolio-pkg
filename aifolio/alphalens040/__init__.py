from . import performance
from . import plotting
from . import tears
from . import utils

from ._version import get_versions


# __version__ = get_versions()['version']
__version__ = '0.4.0'  # 上述方法依赖git tag；本处copy的是0.4.0版本

del get_versions

__all__ = ['performance', 'plotting', 'tears', 'utils']
