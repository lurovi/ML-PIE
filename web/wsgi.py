import os

from numpy import VisibleDeprecationWarning

os.environ['NUMPY_EXPERIMENTAL_ARRAY_FUNCTION'] = '0'
import numpy as np
import warnings
import torch.multiprocessing as mp

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

mp.set_sharing_strategy('file_system')

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

from App import app

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=VisibleDeprecationWarning)
        app.run(threaded=False, processes=mp.cpu_count()-1)
