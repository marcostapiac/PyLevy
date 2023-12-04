import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import project_config
from utils.maths_functions import gammafnc, incgammal, erf, get_z0_H0, hyp1f1

delta = 1.
gamma = 0.5
lambd = .2
mu = 0.
muW = 1.
var_W = 2.
truncations = np.logspace(-10, 0, num=100)
z0, H0 = get_z0_H0(lambd)

tildePi = np.power(np.pi / 2., 0.5)
A1 = 0.5 * np.power(gamma, 2.)
b = A1
fEpsilon = hyp1f1(-1.5, 0.5, -np.power(muW, 2) * np.power(2 * var_W, -1) * truncations)

B1 = 0.7975 * np.power(gamma, 1.5)
r1const = np.power(2.*H0*b, -1)*np.power(delta, -0.5)
rate1 = B1 * r1const * fEpsilon * incgammal(1., b * truncations) * np.power(erf(gamma * np.sqrt(truncations / 2.)), -1.5)

r2const = B1 * np.power(2., np.abs(lambd)) * np.power(delta, 2. * np.abs(lambd) - 1.5) * gammafnc(np.abs(lambd))
r2const *= np.power(np.pi * H0, -1) * np.power(b, 1.5 - np.abs(lambd)) * np.power(z0, 2. * np.abs(
    lambd) - 1.)
rate2 = r2const * fEpsilon * incgammal(1.5 - np.abs(lambd), b * truncations) * np.power(
    erf(gamma * np.sqrt(truncations / 2.)), -1.5)

r3const = B1*np.pi*np.power(b*delta, -1.5)*lambd
rate3 = r3const * fEpsilon * incgammal(1.5, b * truncations)* np.power(erf(gamma * np.sqrt(truncations / 2.)), -1.5)
D1 = b * np.power(gamma, -1.5) * np.power(np.pi, 0.75) * np.power(2., -0.75)

assert (np.abs(B1 * D1*r1const - (0.7975 * np.power(np.pi, 0.75)*np.power(2., -1.75) * np.power(H0, -1) * np.power(delta, -0.5))) < 1e-6)
asymptotic_rate = B1 * D1 * r1const * np.power(truncations, 1 / 4.)
plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.plot(truncations, rate1 + rate2+rate3,label="Bound")
plt.scatter(truncations, asymptotic_rate, marker="v", s=0.5, label="Asymptotic Bound")

plt.title("Bounds on $E_{\epsilon}$ for the GH process")
plt.xlabel("$\epsilon$")
plt.ylabel("Bound on $E_{\epsilon}$")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.savefig(project_config.ROOT_DIR + "/pngs/ConvergenceRateGH.eps",format="eps", dpi=100, bbox_inches="tight")

plt.show()
