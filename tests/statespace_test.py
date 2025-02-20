import matplotlib.pyplot as plt
import numpy as np
from PyLevy.processes.mean_mixture_processes import NormalGammaProcess
from PyLevy.statespace.statespace import LangevinStateSpace

plt.style.use('ggplot')

theta = -.5
initial_state = np.atleast_2d(np.array([0., 0.])).T

observation_matrix1 = np.atleast_2d(np.array([1., 0.]))
observation_matrix2 = np.atleast_2d(np.array([0., 1.]))

alpha = 0.5
beta = 30.
C = 0.02
mu = 0.
mu_W = 0.
var_W = 1.

rng1 = np.random.default_rng(1)
rng2 = np.random.default_rng(1)
rngt = np.random.default_rng(50)

ngp1 = NormalGammaProcess(beta, C, mu, mu_W, var_W, rng=rng1)
# ngp1 = NormalTemperedStableProcess(alpha, beta, C, mu, mu_W, var_W, rng=rng1)
langevin1 = LangevinStateSpace(initial_state, theta, ngp1, observation_matrix1, truncation_level=1e-6, rng=rng1,
                               modelCase=1)
ngp2 = NormalGammaProcess(beta, C, mu, mu_W, var_W, rng=rng2)
# ngp2 = NormalTemperedStableProcess(alpha, beta, C, mu, mu_W, var_W, rng=rng2)
langevin2 = LangevinStateSpace(initial_state, theta, ngp2, observation_matrix2, truncation_level=1e-6, rng=rng2,
                               modelCase=1)
# times = np.random.rand(500).cumsum()
times = rngt.exponential(size=100).cumsum()
xs = langevin1.generate_observations(times, 1e-32)
xdots = langevin2.generate_observations(times, 1e-32)

fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
ax1.plot(times, xs)
ax2.plot(times, xdots)
plt.show()
