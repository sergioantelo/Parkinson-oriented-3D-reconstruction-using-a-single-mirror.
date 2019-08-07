# -*- coding: utf-8 -*-
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


[x1,y1,z1] = (pickle.load(open("coords_horiz", "rb")))
[x2,y2,z2] = (pickle.load(open("coords_transv", "rb")))
[x3,y3,z3] = (pickle.load(open("static", "rb")))

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x1,y1,z1)
ax.plot(x2,y2,z2)
ax.plot(x3,y3,z3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=0, azim=0)
plt.show()

