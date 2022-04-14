import os
import numpy as np
import pdb

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

'''
(x: Any, xp: Any, fp: Any, left: Any | None = None, right: Any | None = None, period: Any | None = None) -> Any
x (corresponding to fp) or ndarray

One-dimensional linear interpolation.

Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x
'''
t1 = np.interp(1, [0, 100], [0, 10])      # 0.1
t2 = np.interp(2, [0, 100], [0, 10])      # 0.2
t3 = np.interp(3, [0, 100], [0, 10])      # 0.30000000000000004
t4 = np.interp(4, [0, 100], [0, 10])      # 0.4
t5 = np.interp(5, [0, 100], [0, 10])      # 0.5
t90 = np.interp(90, [0, 100], [0, 10])    # 9.0

pdb.set_trace()
print('done')
