import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from PyLevy.processes import mean_mixture_processes

plt.style.use('ggplot')

alpha = 1.
beta = 1.
C = 1.

mu = 0.
mu_W = 0.
var_W = 1.

nts = mean_mixture_processes.NormalTemperedStableProcess(alpha, beta, C, mu, mu_W, var_W)
ng = mean_mixture_processes.NormalGammaProcess(beta, C, mu, mu_W, var_W)
axis = np.linspace(0., 1., 1000)
nts_sample = nts.simulate_path(axis)
ng_sample = nts.simulate_path(axis)

fig, [ax1, ax2] = plt.subplots(nrows=1, ncols=2)

ax1.plot(axis, nts_sample, lw=1.2)
ax2.plot(axis, ng_sample, lw=1.2)
plt.show()
