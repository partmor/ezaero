import numpy as np
import matplotlib.pyplot as plt
import ezaero.vlm.steady as vlm
import ezaero.vlm.plotting as plot_utils


WING = vlm.WingParams(1, 0.6, 4, 30 * np.pi / 180, 15 * np.pi / 180)
MESH = vlm.MeshParams(4, 16)
FLCOND = vlm.FlightConditions(100, 1 * np.pi / 180, 1.0)

wing_panels, cpoints = vlm.build_wing_panels(WING, MESH)
vortex_panels = vlm.build_wing_vortex_panels(wing_panels)
normal_vectors = vlm.get_panel_normal_vectors(wing_panels)
surface = vlm.get_wing_planform_surface(wing_panels)
wake = vlm.build_steady_wake(FLCOND, vortex_panels)
aic = vlm.get_influence_matrix(vortex_panels, wake, cpoints, normal_vectors)
rhs = vlm.get_rhs(FLCOND, normal_vectors)
circulation = vlm.solve_net_panel_circulation_distribution(aic, rhs, MESH.m,
                                                           MESH.n)

res = vlm.get_aero_distributions(FLCOND, WING, MESH, circulation, surface)

ax = plot_utils.plot_panels(wing_panels)
plot_utils.plot_panels(vortex_panels, edge_color='r', fill_color=0, ax=ax)
plot_utils.plot_control_points(cpoints, ax)
plt.show()

plot_utils.plot_cl_distribution_on_wing(wing_panels, res)
plt.show()
