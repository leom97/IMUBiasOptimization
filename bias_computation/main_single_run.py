# %% Imports
import matplotlib.pyplot as plt
import logging
import numpy as np

import sys

sys.path.append("/storage/local/mutti/projects/linus_denoising/bias_computation")

from utils.testing import Testing
from utils.data import Data
from utils.optimization import LinearMultilevelOptimization

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

# %% Set-ups

data = Data(dataset="oxiod")
data.get_sequence(mode="train", number=72)   #15
data.build_subsampled_data()
data.build_subsampled_data(rot_cgf={"train": 1, "test": "raw"}, imu_cfg={"train": "raw", "test": "raw"})

optimization = LinearMultilevelOptimization()
optimization.set_data(raw_gyros=data.raw_gyros_train,
                      rotations=data.rotations_train,
                      imu_ts=data.imu_ts_train,
                      gt_ts=data.gt_ts_train)

# %% Get biases

b_opt = optimization.run_optimization(
    {"iterate_tol": 1e-5,
     "maxit": 10,
     "max_step_refinements": 5,
     "levels": 0,  # 0 means no further tree multiplication
     "skip_levels_until": -1,  # -1 means: don't skip anything
     #"loss": {"type": "huber", "huber_weight": 5e-2 * 1 / 200, "maxit": 10},  # mse or huber
     "loss": {"type": "mse"},  # mse, huber (not a good idea) or gaussian
     "smoothing": 0
     }
)

# %% Get errors

testing = Testing()
testing.compute_inference_imu_test_rate(data.rotations_test, data.gt_ts_train, data.raw_gyros_test, data.imu_ts_test,
                                        b_opt)
testing.compute_inference_gt_test_rate(data.imu_ts_test, data.gt_ts_test)
error_gt_test_rate = testing.get_plotting_data(data.rotations_test)

# gt_eul = Rotation.from_matrix(gt_poses_test).as_euler("xyz", degrees=True)
# int_eul = Rotation.from_matrix(R_imu).as_euler("xyz", degrees=True)

# %% Make plots

for i in range(3):
    plt.plot(error_gt_test_rate[:, i])
    plt.title("Error in Euler angles, ground truth rate, " + ["r", "p", "y"][i])
    plt.show()
#
# for i in range(3):
#     plt.plot(gt_timestamps_ns_test, gt_eul[:, i])
#     plt.plot(imu_timestamps_ns_test, int_eul[:, i])
#     plt.title("Absolute trajectories in Euler angles, imu rate, " + ["r", "p", "y"][i])
#     plt.show()
#
# plt.show()
#
plt.plot(b_opt)
plt.title("Biases")
plt.show()

# %% todo
# create DRs already using lietorch!! (for sure, using scipy to get to rotation vectors, instead of getting to quaternions using TLIO, is more than 2x faster
# make the dictionary creation faster!
# use bfgs and not adam!
# introduce a decent interpolation in imu/gt subsampling, instead of what I have now...
# introduce robust loss

# %% Conclusions
# Parking garage will yaw drift a lot, for biases computed with small ground truth windows
# Heavily subsampling ground truth makes for much more accurate biases (maybe the noise in IMU measurements is better averaged out)
# Subsampling IMU makes for faster optimization, but less accurate integrated trajectories
# High frequency training is the only key! Low frequency testing doesn't alter the quality
# Constraining the size of derivatives doesn't work as a smoothing means, for the current cost function! The CURRENT cost function doesn't decrease!
# going nonlinear is a pain (because of setting the correct learning rates in the subproblems) and yields kind of the same results for the non multilevel case (FOR PARKING GARAGE!)
# NB AND FOR NOT PARKING GARAGE????
# it seems that using level = N and skip until N-1, with e.g. 1e5 smoothing on the derivatives, gets better and better result, with increasing N
# However, as it is easy to compute, the determinant for the resulting matrix is zero in this case!
# A multistage smoothing doesn't seem to work either, just like a single stage smoothing

# EXTRA NB: DON'T SKIP IMU MEASUREMENTS WHILE TRAINING, OR YOU WILL GET A LOR OF YAW DRIFT!

# anyhow, subsampling IMUs at test time is still fine
# not using cuda is only 10% faster -> using cpu

# the only going multilevel is confusing!!

# this is probably because of the bad scaling: adding 2**(-level) doesn't make this happen
# conjecture: increasing dt_gt makes now everything worse, which probably means that using upper levels, alone, is too much averaging
# So, this averaging is detrimental, and forcing a multilevel optimization to do this averaging will:
# - if the forcing of this averaging is too large, it will make the optimization worse
# - if the weighting is done right, we will get the same results as in the monolevel case, since the monolevel is very accurate
# Correction: increasing the dt_gt will make things better, but only up to a certain point!