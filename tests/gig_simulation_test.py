import matplotlib.pyplot as plt
import numpy as np
from PyLevy.utils.plotting_functions import plot_path

import project_config
from processes import base_processes

delta = 1.3
gamma = np.sqrt(2.)
lambd = 0.2
nPaths = 10

paths = []
time_ax = np.linspace(0., 1., 1000)
gigp = base_processes.GIGProcess(delta=delta, gamma=gamma, lambd=lambd)

for _ in range(nPaths):
    gigp_sample = gigp.simulate_jumps(truncation=1e-6)
    gigpintegral = gigp.integrate(time_ax, gigp_sample[0], gigp_sample[1])
    paths.append(gigpintegral)

plot_path(time_ax, paths,
          title="Truncated GIG Sample Paths")
plt.savefig(project_config.ROOT_DIR + "/pngs/GIGPathSimulation.eps",format="eps", bbox_inches="tight")

plt.show()
