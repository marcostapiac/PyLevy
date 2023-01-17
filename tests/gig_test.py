import numpy as np
import matplotlib.pyplot as plt
from utils.plotting_functions import qqplot
from processes import base_processes
from tqdm import tqdm

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

title = "Q-Q plot for GIG Process with $\delta, \gamma, \lambda = " + str(delta) + " ," + str(
    round(gamma, 3)) + " ," + str(lambd) + "$"
qqplot(samps, endp, xlabel="True RVs", ylabel="GIG Random Variables at $t = T_{horizon}$", log=True, plottitle=title)
plt.savefig("GIGSimulationQQPlot.png", bbox_inches="tight")
plt.show()
