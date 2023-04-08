import matplotlib.pyplot as plt
import numpy as np
from PyLevy.processes import base_processes
from PyLevy.utils.plotting_functions import plot_path

gamma = np.sqrt(2.)
beta = gamma ** 2 / 2.
nu = 2.
nPaths = 10

paths = []
time_ax = np.linspace(0., 1., 1000)
gp = base_processes.GammaProcess(beta=beta, C=nu)

for _ in range(nPaths):
    gammap_sample = gp.simulate_jumps(truncation=1e-10)
    gpintegral = gp.integrate(time_ax, gammap_sample[0], gammap_sample[1])
    paths.append(gpintegral)

plot_path(time_ax, paths, title="Truncated Gamma Sample Paths")
plt.savefig("../pngs/GammaPathSimulation.png", bbox_inches="tight")
plt.show()
