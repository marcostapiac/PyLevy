import matplotlib.pyplot as plt
import numpy as np
from PyLevy.processes import mean_mixture_processes
from tqdm import tqdm
from PyLevy.utils.maths_functions import normDist
from PyLevy.utils.plotting_functions import qqplot, histogramplot

import project_config

t1 = 0.0
t2 = 1.0
delta = 1.3
gamma = np.sqrt(2.)
lambd = 0.2
mu = 0.
mu_W = 1.
var_W = 2.
truncation = 1e-6

nSamples = 10000

endp = []
gh = mean_mixture_processes.GeneralHyperbolicProcess(delta=delta, gamma=gamma, lambd=lambd, mu=mu, mu_W=mu_W,
                                                     var_W=var_W)

for i in tqdm(range(nSamples)):
    gh_sample = gh.simulate_small_jumps(M=6000, rate=1. / (t2 - t1), truncation=truncation)
    endp.append(np.sum(gh_sample[1]))

endp = np.array(endp)
endp = (endp - np.mean(endp)) / np.std(endp)

rvs = normDist.rvs(size=endp.shape[0])

titleqq = "GH Residual vs Gaussian Distribution"
qqplot(rvs, endp, xlabel="Gaussian Variates", ylabel="Residual GH Variates", plottitle=titleqq, log=False)
plt.savefig(project_config.ROOT_DIR + "/pngs/GHCLTQQ.eps",format="eps", bbox_inches="tight")
plt.show()
plt.close()

hist_axis = np.linspace(normDist.ppf(0.00001), normDist.ppf(0.99999), endp.shape[0])
pdf = normDist.pdf(hist_axis)
titlehist = "Residual GH Density at $t=1$"
#histogramplot(endp, pdf, hist_axis, num_bins=200, xlabel="x", ylabel="Density at $t=1$", plottitle=titlehist)
#plt.savefig(project_config.ROOT_DIR + "/pngs/GHCLTHist.eps",format="eps", bbox_inches="tight")
#plt.show()
