import numpy as np
import matplotlib.pyplot as plt
from PyLevy.utils.plotting_functions import qqplot, histogramplot
from PyLevy.processes import mean_mixture_processes
from PyLevy.utils.maths_functions import normDist, gammafnc
from tqdm import tqdm

t1 = 0.0
t2 = 1.0
kappa = 0.5
gamma = 1.35
delta = 1.
beta = gamma ** (1 / kappa) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1 - kappa))
truncation = 1e-6
mu = 0.
mu_W = 1.
var_W = 2.

nSamples = 100

endp = []
nts = mean_mixture_processes.NormalTemperedStableProcess(alpha=kappa, beta=beta, C=C , mu=mu, mu_W=mu_W, var_W=var_W)

for i in tqdm(range(nSamples)):
    nts_sample = nts.simulate_small_jumps(M=2000, rate=1./(t2-t1), truncation=truncation)
    endp.append(np.sum(nts_sample[1]))

endp = np.array(endp)
endp = (endp - np.mean(endp))/np.std(endp)

rvs = normDist.rvs(size=endp.shape[0])

titleqq = "Q-Q Plot for Residual NTS Process with $\mu, \mu_{W}, \sigma_{W}, \kappa, \gamma, \delta=" + str(mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(kappa) + " ,"+ str(round(gamma,3)) + " ,"+ str(round(delta,3)) + "$"
qqplot(rvs, endp, xlabel="True Normal RVs", ylabel="Residual NTS RVs", plottitle=titleqq, log=False)
plt.savefig("NormalTSCLTQQ.png", bbox_inches = "tight")
plt.show()
plt.close()
hist_axis = np.linspace(normDist.ppf(0.00001), normDist.ppf(0.99999), endp.shape[0])
pdf =normDist.pdf(hist_axis)

titlehist = "Histogram for Residual NTS Process with $\mu, \mu_{W}, \sigma_{W}, \kappa, \gamma, \delta =" + str(mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(kappa) + " ,"+ str(round(gamma,3)) + " ,"+ str(round(delta,3)) + "$"
histogramplot(endp, pdf, hist_axis, num_bins = 200, xlabel="X", ylabel="PDF", plottitle=titlehist)
plt.savefig("NormalTSCLTHist.png", bbox_inches = "tight")
plt.show()