import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
import random

# data
size = 10
p0 = np.random.random((size, 3))
pF = np.random.random((size, 3))

print('p0 = \n', p0)
print('pF = \n', pF)

x0 = p0[:,0].flatten()
y0 = p0[:,1].flatten()
z0 = p0[:,2].flatten()

xF = pF[:,0].flatten()
yF = pF[:,1].flatten()
zF = pF[:,2].flatten()

# initialize 3D plot fig
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# set colormap
cm = plt.get_cmap("RdYlGn")
color = np.linspace(0.0,1.0,size)

# Points (s=point_size, c=color, cmap=colormap)
ax.scatter(x0, y0, z0, s=25, c='blue', marker='o')
ax.scatter(xF, yF, zF, s=25, c='blue', marker='^')  


# Edges
for i in range(size):
	xe = [x0[i], xF[i]]
	ye = [y0[i], yF[i]]
	ze = [z0[i], zF[i]]
	ax.plot(xe, ye, ze, c='red')


ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.set_zlim([-0.1, 1.1])


plt.show()