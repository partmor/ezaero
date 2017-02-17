# aeroelasticipy

This project is a *re-edition* of my Master Thesis, done back in 2013/2014. My MT covered the analysis of the **3D aeroelastic response of an UAV** in a gust scenario.

This project was coded back then in Matlab, and was my first step into *serious* engineering programming. I had a lot of fun during the project, and now I revisit the memory chest to migrate the code to Python 3.x.

The objective is mainly to gain further numpy skills. I thought this is a good exercise; as a Data Scientist, I do work a lot with Pandas, but not so much directly with numpy arrays - *but Pedro, pandas dataframes are in essence 2D numpy arrays bla bla* - but **you know what I mean**, I'm talking about *fancy numpy*, *rocket science numpy*.

Furthermore, I was also curious about how could I improve the code performance (using less naive constructions, parallelizing if possible, and have an excuse to also look through MinPy, to send my heavy numpy operations to GPU rather than CPU).

I will first walk through the 3D steady VLM, then unsteady 3D VLM, and finally construct the free body motion equations calculating the aerodynamic forces with the VLM module.

I will be working on this occasionally as a **hobby**, so I guess my contributions will be sparse in time.

If you somehow landed here, welcome. Feedback appreciated :)

### Fun fact 1:
I have managed to suppress all the for loops in the influence matrix and rhs term construction (all the Biot-Savart law evaluations) relying on **numpy broadcasting**, cross and dot products with high order arrays (the latter by means of np.einsum), improving execution time around x150!, compared to using numpy with naive for loop constructions.

### Milestone 1:
The steady 3D Vortex Lattice Method module is ready and validated against the figures in *Low Speed Aerodynamics* (J. Katz et al.). Only minor improvements will be made, related to controlling the access to some attributes and methods of the class. Now I can start working on  the unsteady VLM.
