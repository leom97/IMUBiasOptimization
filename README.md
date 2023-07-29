# IMUBiasOptimization

This Python code serves to find the best gyroscopic biases in a window of IMU measurements $\omega_1, ..., \omega_N$.

In particular, given 3D orientations of $R_1,...,R_N$ for the window, we aim at finding $b_1,...,b_N$ such that the following metric is minimized:

$$|R_N^TR_1\log(\Pi_{k=1}^N Exp(\omega_k \delta t))|$$

The code provides interfaces to the Euroc and FourSeasons dataset (see https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets and https://www.4seasons-dataset.com/ respectively). 

## Instructions

After cloning the repository, the user can go ahead and run `bias_computation/main_single_run.py`.
