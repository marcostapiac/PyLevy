import numpy as np
import matplotlib.pyplot as plt
from PyLevy.utils.plotting_functions import qqplot, histogramplot
from PyLevy.processes import mean_mixture_processes
from PyLevy.utils.maths_functions import normDist
from tqdm import tqdm

t1 = 0.0
t2 = 1.0
gamma = np.sqrt(2.)
beta = gamma**2/2.
nu = 2.
mu = 0.
mu_W = 1.
var_W = 2.
truncation = 1e-6
nSamples = 100000

endp = []
ng = mean_mixture_processes.NormalGammaProcess(beta=beta, C=nu, mu=mu, mu_W=mu_W, var_W=var_W)

for i in tqdm(range(nSamples)):
    g_sample = ng.simulate_small_jumps(M=2000, rate=1./(t2-t1), truncation=truncation)
    endp.append(np.sum(g_sample[1]))

endp = np.array(endp)
endp = (endp - np.mean(endp))/np.std(endp)

rvs = normDist.rvs(size=endp.shape[0])

pgf = True
titleqq = "Q-Q Plot for Residual NG Process with $\mu, \mu_{W}, \sigma_{W}, \\nu, \gamma =" + str(mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(nu) + " ,"+ str(round(gamma,3)) +"$"
qqplot(rvs, endp, xlabel="True Normal RVs", ylabel="Residual NG RVs", plottitle=titleqq, log=False)
plt.savefig("NormalGammaCLTQQ.png", bbox_inches = "tight")
plt.show()
plt.close()

hist_axis = np.linspace(normDist.ppf(0.00001), normDist.ppf(0.99999), endp.shape[0])
pdf =normDist.pdf(hist_axis)

titlehist = "Histogram for Residual NG Process with $\mu, \mu_{W}, \sigma_{W}, \\nu, \gamma =" + str(mu) + " ," + str(mu_W) + " ," + str(var_W) + " ," + str(nu) + " ,"+ str(round(gamma,3)) +"$"
histogramplot(endp, pdf, hist_axis, xlabel="X", ylabel="PDF", plottitle=titlehist)
plt.savefig("NormalGammaCLTHist.png", bbox_inches = "tight")
plt.show()