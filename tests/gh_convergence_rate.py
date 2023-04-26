import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utils.maths_functions import gammafnc, incgammal, erf, get_z0_H0, hyp1f1

delta = 1.
gamma = 0.5
lambd = -.2
mu = 0.
muW = 1.
var_W = 2.
truncations = np.logspace(-20, 0, num=1000000)
z0, H0 = get_z0_H0(lambd)

tildePi = np.power(np.pi / 2., 0.5)
A1 = 0.5 * np.power(gamma, 2.)
b = A1
fEpsilon = hyp1f1(-1.5, 0.5, -np.power(muW, 2) * np.power(2 * var_W, -1) * truncations)

B1 = 0.7975 * 4. * np.power(gamma, 1.5)
B1 *= np.power(b * H0, -1) * np.power(np.pi, -2) * np.power(delta, -0.5)
rate1 = B1 * fEpsilon * incgammal(1., b * truncations) * np.power(erf(gamma * np.sqrt(truncations / 2.)), -1.5)

C1 = 0.7975 * np.power(2., np.abs(lambd) + 1) * np.power(delta, 2. * np.abs(lambd) - 1.5) * gammafnc(np.abs(lambd))
C1 *= np.power(gamma, 1.5)
C1 *= np.power(tildePi * H0, -1) * np.power(np.pi, -2) * np.power(b, 1.5 - np.abs(lambd)) * np.power(z0, 2. * np.abs(
    lambd) - 1.)
rate2 = C1 * fEpsilon * incgammal(1.5 - np.abs(lambd), b * truncations) * np.power(
    erf(gamma * np.sqrt(truncations / 2.)), -1.5)

D1 = b * np.power(2 * gamma, -1.5) * np.power(np.pi * 2, 0.75)
assert (np.abs(B1 * D1 - (0.7975 * np.power(2. / np.pi, 1.25) * np.power(H0, -1) * np.power(delta, -0.5))) < 1e-6)
asymptotic_rate = B1 * D1 * np.power(truncations, 1 / 4.)
plt.style.use('ggplot')
matplotlib.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
plt.plot(truncations, rate1 + rate2, label="Bound")
plt.plot(truncations, asymptotic_rate, label="Asymptotic Bound")

plt.title("Bounds on $E_{\epsilon}$ for the GH process")
plt.xlabel("$\epsilon$")
plt.ylabel("Bound on $E_{\epsilon}$")
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.savefig("../pngs/ConvergenceRateGH.png")
plt.show()
