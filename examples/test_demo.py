import numpy as np
import matplotlib.pyplot as plt
import ezaero.vlm.steady as svlm
import ezaero.vlm.plotting as plot_utils

WING = svlm.WingParams(1, 1 ,10 ,0 , 25 * np.pi / 180)
MESH = svlm.MeshParams(2, 6)

panels = svlm.build_wing_panels(WING, MESH)
plot_utils.plot_panels(WING, MESH, panels[0])
plt.show()
