import numpy as np
import matplotlib.pyplot as plt
import ezaero.vlm.steady as svlm
import ezaero.vlm.plotting as plot_utils

# WING = svlm.WingParams(1, 1 , 10, 0, 25 * np.pi / 180)
WING = svlm.WingParams(1, 0.6 , 4, 30 * np.pi / 180, 25 * np.pi / 180)
MESH = svlm.MeshParams(4, 16)

wing_panels, cpoints = svlm.build_wing_panels(WING, MESH)
vortex_panels = svlm.build_wing_vortex_panels(MESH, wing_panels)

ax = plot_utils.plot_panels(wing_panels)
plot_utils.plot_panels(vortex_panels, edge_color='r', fill_color=0, ax=ax)
plt.show()
