import matplotlib.pyplot as plt
import numpy as np
from PyLevy.utils.plotting_functions import qqplot
from tqdm import tqdm

import project_config
from processes import base_processes

delta = 1.3
gamma = np.sqrt(2.)
lambd = .2
nSamples = 40000

endp = []
gig = base_processes.GIGProcess(delta=delta, gamma=gamma, lambd=lambd)
samps = gig.generate_marginal_samples(numSamples=nSamples)

axis = np.linspace(0., 1., nSamples)
for i in tqdm(range(nSamples)):
    gig_sample = gig.simulate_jumps(M=6000, truncation=1e-6)
    endpoint = np.sum(gig_sample[1])
    endp.append(endpoint)

title = "Truncated vs true GIG density at $t=1$"
qqplot(samps, endp, xlabel="True GIG Variates", ylabel="Truncated GIG Variates", log=True, plottitle=title)
plt.savefig(project_config.ROOT_DIR + "/pngs/GIGSimulationQQPlot.eps",format="eps", bbox_inches="tight")
plt.show()
