import matplotlib.pyplot as plt
import numpy as np
from PyLevy.processes import base_processes
from PyLevy.utils.maths_functions import gammafnc
from PyLevy.utils.plotting_functions import plot_path

kappa = 0.5
gamma = 1.35
delta = 1.
beta = gamma ** (1 / kappa) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))
nPaths = 10

paths = []
time_ax = np.linspace(0., 1., 1000)
tsp = base_processes.TemperedStableProcess(alpha=kappa, beta=beta, C=C)

for _ in range(nPaths):
    tsp_sample = tsp.simulate_jumps(truncation=1e-10)
    tspintegral = tsp.integrate(time_ax, tsp_sample[0], tsp_sample[1])
    paths.append(tspintegral)

plot_path(time_ax, paths, title="Truncated Tempered Stable Sample Paths")
plt.savefig("../pngs/TSPathSimulation.png", bbox_inches="tight")
plt.show()
