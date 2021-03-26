#!/usr/bin/env python3
import numpy as np
from scipy.spatial.transform import Rotation as R

vec = np.random.rand(3)

for i in range(100):
    q = np.random.rand(4) - 0.5
    rot = R.from_quat(q)
    vec2 = rot.apply(vec)
    x, y, z = vec2
    print(
        x ** 4
        + y ** 4
        + z ** 4
        + 2 * (x ** 2 * y ** 2 + x ** 2 * z ** 2 + y ** 2 * z ** 2)
    )
