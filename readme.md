This is the code for the ICLR 2023 Submission, 'Actionable Neural Representations: Grid Cells from Minimal Constraints'.

Included in the code are:

1. 6 files that perform optimisation in finite periodic of very large periodic spaces of 1, 2, or 3 dimensions
Line, Circle, Torus, Plane, Volume, PeriodicVolume.py
These should just run and output saved lists of the relevant parameters which can be used for plotting

2. A folder which acts as a module, NRT_functions (NRT = Neural Representation Theory), which contains three files
helper_functions.py - just some helpful functions
losses.py - the losses used
plotter.py - some helpful plotting functions

3. A jupyter notebook that lets you plot the output of a simulation
Results.ipynb


Required packages:
1. Jax
2. Numpy
3. matplotlib
