import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PyLevy.utils.maths_functions import kstest, normDist, gammafnc, incgammal
from PyLevy.utils.plotting_functions import qqplot, histogramplot
from tqdm import tqdm

muW = 1.
var_W = 2.
kappa = .5
gamma = 1.35
delta = 1.
beta = gamma ** (1 / kappa) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1. - kappa))
truncations = np.arange(1e-6, 1e-2, step=1e-6)
mu = 0.

A1 = 0.5 * np.power(gamma, 1. / kappa)
A2 = A1 + np.power(muW, 2) * np.power(2 * var_W, -1)
B1 = 0.7975 * np.power(2., kappa)
B1 *= np.power(delta * kappa * np.pi * gammafnc(1. - kappa), -0.5)
B1 *= np.power(gamma, -(3. * kappa - 3.) / (2. * kappa))
B1 *= np.power(A2, kappa - 1.5)
rate = B1 * incgammal(1.5 - kappa, A2 * truncations) * np.power(
    incgammal(1. - kappa, A1 * truncations),
    -1.5)

C1 = np.power(A2, 1.5 - kappa) * np.power(1.5 - kappa, -1) * np.power(1. - kappa, 1.5) * np.power(A1, -1.5)

asymptotic_rate = B1 * C1 * np.power(truncations, kappa / 2.)
plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.plot(truncations, rate, label="Rate")
plt.plot(truncations, asymptotic_rate, label="Asymptotic Rate")
plt.title("Convergence Rate for NTS Process")
plt.xlabel("Truncation Level, $\epsilon$")
plt.ylabel("Convergence Rate")
plt.legend()
plt.savefig("ConvergenceRateNTS.png")
plt.show()
