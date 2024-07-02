# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os

BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

import numpy as np
import pc_util
from plyfile import PlyData, PlyElement

filename = sys.argv[1]
assert os.path.isfile(filename)
with open(filename, "rb") as f:
    plydata = PlyData.read(f)
    num_verts = plydata["vertex"].count
    for i in range(0,num_verts):
        print("1",end=',')
    vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
    vertices[:, 0] = plydata["vertex"].data["x"]
    vertices[:, 1] = plydata["vertex"].data["y"]
    vertices[:, 2] = plydata["vertex"].data["z"]
    vertices[:, 3] = plydata["vertex"].data["red"]
    vertices[:, 4] = plydata["vertex"].data["green"]
    vertices[:, 5] = plydata["vertex"].data["blue"]

