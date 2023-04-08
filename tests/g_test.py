import matplotlib.pyplot as plt
import numpy as np
from PyLevy.processes import base_processes
from PyLevy.utils.maths_functions import gammaDist
from PyLevy.utils.plotting_functions import histogramplot
from tqdm import tqdm

t1 = 0.0
t2 = 1.0
gamma = np.sqrt(2.)
beta = gamma ** 2 / 2.
nu = 2.
nSamples = 100000

endp = []
g = base_processes.GammaProcess(beta=beta, C=nu)

for i in tqdm(range(nSamples)):
    g_sample = g.simulate_jumps(M=2000, rate=1. / (t2 - t1), truncation=1e-10)
    endp.append(np.sum(g_sample[1]))

pdf = gammaDist.pdf(x=np.linspace(min(endp), max(endp), len(endp)), a=nu, loc=0., scale=1 / beta)

histogramTitle = "Truncated vs true Gamma density at $t=1$"
histogramplot(rvs=endp, pdf_vals=pdf, axis=np.linspace(min(endp), max(endp), len(endp)), xlabel="$x$",
              ylabel="Density at $t=1$", plottitle=histogramTitle)

plt.savefig("../pngs/GammaPathHistogram.png", bbox_inches="tight")
plt.show()
