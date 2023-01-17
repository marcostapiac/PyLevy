import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PyLevy.utils.maths_functions import gammafnc
from PyLevy.utils.plotting_functions import qqplot
from PyLevy.processes import base_processes


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
    _, ts_sample = ts.simulate_jumps(M=2000,  truncation=1e-10)
    endp.append(np.sum(ts_sample))
samps = ts.generate_marginal_samples(numSamples=nSamples, tHorizon=t2-t1)

title = "Q-Q plot for TS Process with $\kappa, \gamma, \delta = " + str(kappa)+" ,"+ str(round(gamma, 3)) + " ," + str(delta) + "$"
qqplot(samps, endp, xlabel="True RVs", ylabel="TS Random Variables at $t = T_{horizon}$", plottitle=title)
plt.savefig("TSSimulationQQPlot.png", bbox_inches = "tight")
plt.show()