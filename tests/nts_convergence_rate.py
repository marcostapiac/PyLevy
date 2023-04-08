import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyLevy.utils.maths_functions import gammafnc, incgammal, hyp1f1

muW = 1.
var_W = 2.
kappa = .5
gamma = 1.35
delta = 1.
beta = (gamma ** (1 / kappa)) / 2.0
C = delta * (2 ** kappa) * kappa * (1 / gammafnc(1. - kappa))
truncations = np.arange(1e-6, 1e-0, step=1e-6)

A2 = 0.5 * np.power(gamma, 1. / kappa)
B1 = 0.7975 * np.power(2., 1.5) * np.power(gammafnc(1. - kappa), 0.5)
B1 *= np.power(delta * kappa * gamma * np.pi, -0.5)
fEpsilon = hyp1f1(-1.5, 0.5, -np.power(muW, 2) * np.power(2 * var_W, -1) * truncations)
rate = B1 * incgammal(1.5 - kappa, A2 * truncations) * np.power(
    incgammal(1. - kappa, A2 * truncations),
    -1.5)

C1 = np.power(A2, 1.5 - kappa) * np.power(1.5 - kappa, -1) * np.power(1. - kappa, 1.5) * np.power(A2,
                                                                                                  -1.5 + 1.5 * kappa)

asymptotic_rate = B1 * C1 * np.power(truncations, kappa / 2.)
plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.plot(truncations, rate, label="Rate")
plt.plot(truncations, asymptotic_rate, label="Asymptotic Rate")
plt.title("Bounds on $E_{\epsilon}$ for the NTS process")
plt.xlabel("$\epsilon$")
plt.ylabel("Bound on $E_{\epsilon}$")
plt.legend()
plt.savefig("../pngs/ConvergenceRateNTS.png")
plt.show()
