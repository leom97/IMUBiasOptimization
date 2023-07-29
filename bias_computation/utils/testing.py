from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import numpy as np


class Testing():

    def __init__(self):
        self.gyro_debiased = None
        self.rotations_inference_gt_test_rate = None
        self.rotations_inference_imu_test_rate = None

    def compute_inference_imu_test_rate(self, rotations_test, gt_ts_train, raw_gyros_test, imu_ts_test, b_opt):
        biases = interp1d(gt_ts_train[:-1], b_opt, kind="previous", fill_value="extrapolate", axis=0)(imu_ts_test)
        self.gyro_debiased = raw_gyros_test - biases

        # result
        R_imu = [Rotation.from_matrix(rotations_test[0])]
        dR_imu = Rotation.from_rotvec(self.gyro_debiased[:-1] * np.diff(imu_ts_test)[:, None])

        for dR_imu_tmp in dR_imu:
            R_imu.append(R_imu[-1] * dR_imu_tmp)

        self.rotations_inference_imu_test_rate = np.concatenate([Rotation.as_matrix(R)[None, :] for R in R_imu], axis=0)

    def compute_inference_gt_test_rate(self, imu_ts_test, gt_ts_test):
        self.rotations_inference_gt_test_rate = interp1d(imu_ts_test, self.rotations_inference_imu_test_rate,
                                                         kind="nearest", fill_value="extrapolate", axis=0)(
            gt_ts_test)  # subsampled to gt rate

    def get_plotting_data(self, rotations_test):
        error_gt_test_rate = Rotation.from_matrix(
            np.einsum("bij, bkj -> bik", rotations_test, self.rotations_inference_gt_test_rate)).as_euler("xyz",
                                                                                                          degrees=True)

        return error_gt_test_rate
