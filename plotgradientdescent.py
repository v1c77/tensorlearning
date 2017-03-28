
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plot
import numpy as np

fig = plot.figure()
ax = fig.gca(projection='3d')

s = 0.05  # Try s=1, 0.25, 0.1, or 0.05
X = np.arange(-2, 2. + s, s)  # Could use linspace instead if dividing
Y = np.arange(-2, 3. + s, s)  # evenly instead of stepping...

# Create the mesh grid(s) for all X/Y combos.
X, Y = np.meshgrid(X, Y)

# Rosenbrock function w/ two parameters using numpy Arrays
Z = (1. - X) ** 2 + 100. * (Y - X * X) ** 2

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)  # Try coolwarm vs jet

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

# Displays the figure, handles user interface, returns when user closes window
plot.show()