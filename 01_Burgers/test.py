"""
********************************************************************************
test to visualize reference solution to 1D Burgers equation
********************************************************************************
"""

import numpy as np
from scipy import io
import matplotlib.pyplot as plt

# load data
data = io.loadmat("./ref_sol.mat")
t = data["t"].flatten()[:,None]
x = data["x"].flatten()[:,None]

# shape before meshgrid
print("t.shape:", t.shape)
print("x.shape:", x.shape)

t, x = np.meshgrid(t, x)
u_ref = data["usol"]
u_ref = np.real(u_ref)

# shape after meshgrid
print("t.shape:", t.shape)
print("x.shape:", x.shape)

# bound?
print("tmin, tmax:", t.min(), t.max())
print("xmin, xmax:", x.min(), x.max())

# visualize
plt.figure(figsize=(8, 4))
plt.scatter(t, x, s=1.)
plt.scatter(t, x, c=u_ref, cmap="turbo", vmin=-1., vmax=1.)
plt.colorbar(ticks=np.arange(-1., 1.1, .5))
plt.xticks(np.arange(t.min(), t.max()+.1, .25))
plt.yticks(np.arange(x.min(), x.max()+.1, .5))
plt.xlim(t.min(), t.max())
plt.ylim(x.min(), x.max())
plt.xlabel("t")
plt.ylabel("x")
plt.show()
