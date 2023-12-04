import matplotlib.pyplot as plt
import numpy as np
from PyLevy.processes import mean_mixture_processes
from PyLevy.utils.maths_functions import normDist
from PyLevy.utils.plotting_functions import qqplot, histogramplot
from tqdm import tqdm

import project_config

t1 = 0.0
t2 = 1.0
gamma = np.sqrt(2.)
beta = gamma ** 2 / 2.
nu = 2.
mu = 0.
mu_W = 1.
var_W = 2.
truncation = 1e-6
nSamples = 100000

endp = []
ng = mean_mixture_processes.NormalGammaProcess(beta=beta, C=nu, mu=mu, mu_W=mu_W, var_W=var_W)

for i in tqdm(range(nSamples)):
    g_sample = ng.simulate_small_jumps(M=20000, rate=1. / (t2 - t1), truncation=truncation)
    endp.append(np.sum(g_sample[1]))

endp = np.array(endp)
endp = (endp - np.mean(endp)) / np.std(endp)

rvs = normDist.rvs(size=endp.shape[0])

pgf = True
titleqq = "NG Residual vs Gaussian Distribution"
qqplot(rvs, endp, xlabel="Gaussian Variates", ylabel="NG Residual Variates", plottitle=titleqq, log=False)
plt.savefig(project_config.ROOT_DIR + "/pngs/NormalGammaCLTQQ.eps",format="eps", bbox_inches="tight")
plt.show()
plt.close()

hist_axis = np.linspace(normDist.ppf(0.00001), normDist.ppf(0.99999), endp.shape[0])
pdf = normDist.pdf(hist_axis)

titlehist = "Residual NG Density at $t=1$"
histogramplot(endp, pdf, hist_axis, xlabel="x", ylabel="Density at $t=1$", plottitle=titlehist)
plt.savefig(project_config.ROOT_DIR + "/pngs/NormalGammaCLTHist.eps",format="eps", bbox_inches="tight")
plt.show()
