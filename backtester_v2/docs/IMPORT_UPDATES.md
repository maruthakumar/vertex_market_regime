
# Import Updates Required After Reorganization

## For files that import from BTRUN:

### Old imports:
```python
from . import config
from . import runtime
from . import io
```

### New imports:
```python
from .core import config
from .core import runtime
from .core import io
```

## For the main GPU runners (BTRunPortfolio_GPU.py, BT_TV_GPU.py, BT_OI_GPU.py):

### Old imports:
```python
from backtester_stable.BTRUN import config, utils, runtime
from backtester_stable.BTRUN import gpu_helpers, io, stats
```

### New imports:
```python
from backtester_stable.BTRUN.core import config, utils, runtime
from backtester_stable.BTRUN.core import gpu_helpers, io, stats
```

## Update __init__.py files:

Create/update __init__.py in each subdirectory to export the modules properly.
