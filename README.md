# aeroelasticipy

This project is a *re-edition* of my Master Thesis, done back in 2013/2014. My MT covered the analysis of the **3D aeroelastic response of an UAV** in a gust scenario.

This project was originally coded in Matlab. Now, I revisit the memory chest to migrate the code to Python 3.x, mainly because I was very curious about how could I improve the code performance, and also be able to share my results with the community.

I will first walk through the 3D steady VLM, 3D then unsteady VLM, and finally construct the free body motion equations calculating the aerodynamic forces with the VLM module.

+ Steady 3D VLM [demo](https://github.com/partmor/aeroelasticipy/blob/master/steady_VLM/demo.ipynb)

### Milestones:
+ The steady 3D Vortex Lattice Method module is ready and validated against the figures in *Low Speed Aerodynamics* (J. Katz et al.). Only minor further improvements will be made, related to controlling the access to some attributes and methods of the class. Now I can start working on the unsteady VLM.

#### Misc:
+ **Steady VLM**: I have managed to suppress all the for loops in the influence matrix and rhs term construction (all the Biot-Savart law evaluations) relying on **numpy broadcasting**, cross and dot products with high order arrays (the latter by means of np.einsum), improving execution time around x150!, compared to using numpy with naive for loop constructions.
