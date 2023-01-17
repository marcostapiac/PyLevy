import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PyLevy.utils.plotting_functions import qqplot, histogramplot
from PyLevy.utils.maths_functions import kstest, normDist, gammafnc, incgammal, erf, get_z0_H0
from tqdm import tqdm

delta = 1.
gamma = 0.5
lambd = -.2
mu = 0.
muW = 1.
var_W = 2.
truncations = np.arange(1e-6, 1e-2, step=1e-6)
z0, H0 = get_z0_H0(lambd)

tildePi = np.power(np.pi / 2., 0.5)
A1 = 0.5 * np.power(gamma, 2.)
A2 = A1 + np.power(muW, 2) * np.power(2 * var_W, -1)

B1 = 4. * gamma * np.power(gamma, 0.5) * np.power(var_W, 1.5)
B1 *= np.power(A2 * H0, -1) * np.power(np.pi, -2) * np.power(delta, -0.5)
rate1 = B1 * incgammal(1., A2 * truncations) * np.power(erf(gamma * np.sqrt(truncations / 2.)), -1.5)

C1 = np.power(2., np.abs(lambd) + 1) * np.power(var_W, 1.5) * np.power(delta, 2. * np.abs(lambd) - 1.5) * gammafnc(
    np.abs(lambd))
C1 *= gamma * np.power(gamma, 0.5)
C1 *= np.power(tildePi * H0, -1) * np.power(np.pi, -2) * np.power(A2, 1.5 - np.abs(lambd)) * np.power(z0, 2. * np.abs(
    lambd) - 1.)
rate2 = C1 * incgammal(1.5 - np.abs(lambd), A2 * truncations) * np.power(erf(gamma * np.sqrt(truncations / 2.)), -1.5)

D1 = np.power(np.pi, 0.75) * A2
D1 *= np.power(gamma * np.sqrt(2), -1.5)
asymptotic_rate = B1 * D1 * np.power(truncations, 1 / 4.)
plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.plot(truncations, rate1 + rate2, label="Rate")
plt.plot(truncations, asymptotic_rate, label="Asymptotic Rate")
plt.title("Convergence Rate for GH Process")
plt.xlabel("Truncation Level, $\epsilon$")
plt.ylabel("Convergence Rate")
plt.legend()
plt.savefig("ConvergenceRateGH.png")
plt.show()
