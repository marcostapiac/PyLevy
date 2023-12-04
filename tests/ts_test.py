import matplotlib.pyplot as plt
import numpy as np
from PyLevy.processes import base_processes
from PyLevy.utils.maths_functions import gammafnc
from PyLevy.utils.plotting_functions import qqplot
from tqdm import tqdm

import project_config

t1 = 0.0
t2 = 1.0
kappa = 0.5
gamma = 1.35
delta = 1.
beta = gamma ** (1 / kappa) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))

ts = base_processes.TemperedStableProcess(alpha=kappa, beta=beta, C=C)
nSamples = 100000
endp = []

for i in tqdm(range(nSamples)):
    _, ts_sample = ts.simulate_jumps(M=2000, truncation=1e-10)
    endp.append(np.sum(ts_sample))
samps = ts.generate_marginal_samples(numSamples=nSamples, tHorizon=t2 - t1)

title = "Truncated vs true Tempered Stable density at $t=1$"
qqplot(samps, endp, xlabel="True Tempered Stable Variates", ylabel="Truncated Tempered Stable Variates",
       plottitle=title)
plt.savefig(project_config.ROOT_DIR + "/pngs/TSSimulationQQPlot.eps",format="eps", bbox_inches="tight")
plt.show()
