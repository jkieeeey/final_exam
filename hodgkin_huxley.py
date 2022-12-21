#%%
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import math

#%%
def hodgkinhuxley(state,t):
	v = state[0]
	y = state[1]
	m = state[2]
	h = state[3]
	a_n = 0.01 * (10 - v) / (np.exp((10 - v) / 10) - 1)
	b_n = 0.125 * np.exp( - v / 80)
	a_m = 0.1 * (25 - v) / (np.exp((25 - v) / 10) - 1)
	b_m = 4 * np.exp( - v / 18)
	b_h = 1 / (np.exp((30 - v) / 10) + 1)
	a_h = 0.07 * np.exp( - v / 20)
	v_Na = 115
	v_K = -12
	v_L = 10.6
	gbar_Na = 120
	g_Na = gbar_Na * h * m ** 3
	gbar_K = 36
	g_K = gbar_K * (y ** 4)
	gbar_L = 0.3
	c_m = 1

	i_Na = g_Na * (v - v_Na)
	i_K = g_K * (v - v_K)
	i_L = gbar_L
	i_app = i_in
	dvdt = (- i_Na - i_K - i_L + i_app) / c_m
	dndt = a_n * (1 - y) - b_n * y
	dmdt = a_m * (1 - m) - b_m * m
	dhdt = a_h * (1 - h) - b_h * h
	return dvdt, dndt, dmdt, dhdt

#%%
i_table = [1.0, 2.9, 3.3, 3.5]
t = np.arange (0, 60, 0.001)
state0 = [0, 0.28, 0.05, 0.6]
for i_in in i_table:
	state = odeint(hodgkinhuxley, state0, t)
	plt.plot(t, state[:, 0] - 65)
	plt.show()