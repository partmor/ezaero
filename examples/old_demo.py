import numpy as np
import matplotlib.pyplot as plt
import ezaero.vlm.steady as svlm

bp = 4
m, n = 4, 16
T = 30 * np.pi / 180
delta = 10 * np.pi / 180
c_r, c_t = 1, 0.6
U_i = 100
rho = 1.0
alpha = 1 * np.pi / 180

sim = svlm.Steady_VLM()

sim.set_wing_parameters(bp,T,delta,c_r,c_t)
sim.set_mesh_parameters(m,n)
sim.set_flight_conditions(U_i,alpha,rho)

sim.plot_mesh()
plt.show()

sim.run()
sim.plot_cl_distribution_on_wing()
plt.show()

sim.plot_normalized_spanwise_cl()
plt.show()
