import logging
logging.getLogger().setLevel(logging.INFO)

import torch

from geometry_utils.se3 import Se3
from geometry_utils import so3
from geometry_utils import imu
from dataloading.sequence import Sequence

from scipy.interpolate import interp1d
import numpy
import os
from pathlib import Path
from libartipy.geometry import Pose
from scipy.spatial.transform import Rotation
import h5py
import numpy as np


class Constants:
    gt_filename = 'data.hdf5'
    imu_filename = 'imu.txt'
    transformations_filename = 'Transformations.txt'


def interpolate_poses(poses: Se3, pose_timestamps, target_timestamps):
    r_interpolated = so3.interpolate_quaternions(poses.q, pose_timestamps, target_timestamps).R
    t_interpolated = interp1d(pose_timestamps, poses.t, axis=0)(target_timestamps)
    gt_poses = Se3(r_interpolated, t_interpolated)
    gt_velocities = (t_interpolated[1:] - t_interpolated[:-1]) / numpy.diff(target_timestamps * 1e-9)[:, None]
    return gt_poses, gt_velocities


def load_sequence(sequence_dir, subsampling_step, do_bias_correction=True):
    logging.info(f"Loading {sequence_dir}")
    gt_poses, gt_timestamps, gyro_h5py, acce_h5py = load_gt(os.path.join(sequence_dir, Constants.gt_filename), do_bias_correction=do_bias_correction)
    imu_measurements, imu_timestamps = load_imu(sequence_dir, gt_timestamps[0], gt_timestamps[-1], gyro_h5py=gyro_h5py,
                                                acce_h5py=acce_h5py, ts_h5py=gt_timestamps)

    gt_poses_interpolated, gt_velocities = interpolate_poses(gt_poses, gt_timestamps,
                                                             imu_timestamps[::subsampling_step])

    # now, the ground truth biases (if we were to integrate everything)
    vio_R = Rotation.from_matrix(gt_poses_interpolated.R[:len(imu_timestamps)])
    dts = np.diff(imu_timestamps[::subsampling_step])
    bias_gt = imu_measurements[::subsampling_step][:-1, :3] - (vio_R[:-1].inv() * vio_R[1:]).as_rotvec() / dts[:, None]
    gyro_bias_gt = numpy.append(bias_gt, bias_gt[-1][None, :], axis=0)

    return Sequence(
        poses=gt_poses_interpolated,
        velocities=gt_velocities,
        imu_measurements=imu_measurements[::subsampling_step],
        timestamps=imu_timestamps[::subsampling_step],
        gyro_bias_gt=gyro_bias_gt
    )

def load_gt(npz_path, do_bias_correction=False):
    with h5py.File(npz_path, "r") as f:
        ts = np.copy(f["ts"])
        vio_q = np.copy(f["vio_q_wxyz"])
        vio_p = np.copy(f["vio_p"])
        if do_bias_correction:
            gyro = np.copy(f["gyro_dcalibrated"])
            accel = np.copy(f["accel_dcalibrated"])
        else:
            gyro = np.copy(f["gyro_raw"])
            accel = np.copy(f["accel_raw"])
    # return Se3(npz_poses["R"].astype(numpy.float64), npz_poses["t"].astype(numpy.float64)), npz_poses["timestamps"]
    return Se3(Rotation.from_quat(vio_q[:, [1, 2, 3, 0]]).as_matrix().astype(numpy.float64),
               vio_p.astype(numpy.float64)), ts, gyro, accel


def load_imu(sequence_dir, start_timestamp, end_timestamp, gyro_h5py=None, acce_h5py=None, ts_h5py=None):
    """Load imu data from file. Ignoring datapoints before start_timestamp"""

    if gyro_h5py is None or acce_h5py is None or ts_h5py is None:
        imu_data = numpy.loadtxt(os.path.join(sequence_dir, Constants.imu_filename), delimiter=" ")
        imu_timestamps = imu_data[:, 0].astype(numpy.int64)
        imu_measurements = imu_data[:, 1:]
    else:
        imu_timestamps = ts_h5py
        imu_measurements = np.concatenate((gyro_h5py, acce_h5py), axis=1)

    # transforms = load_transforms(os.path.join(sequence_dir, Constants.transformations_filename))
    # imu_measurements = imu.rotate_imu(imu_measurements, transforms["TS_cam_imu"].rotation_matrix.T)

    start_index = numpy.searchsorted(imu_timestamps, start_timestamp)
    end_index = numpy.searchsorted(imu_timestamps, end_timestamp) - 1
    return imu_measurements[start_index:end_index], imu_timestamps[start_index:end_index]


def load_transforms(transforms_file):
    with open(transforms_file, 'r') as transformations_file:
        lines = transformations_file.readlines()
    transforms_dict = {}
    for i in range(len(lines)):
        if lines[i][0] == "#" and lines[i] != "# GNSS scale\n":
            name = lines[i][2:].split(':')[0]
            transforms_dict[name] = Pose.from_line(lines[i + 1].rstrip('\n'))
    return transforms_dict


class DatasetDefinitions:
    imu_frequency = 2000.0
    train_sequences = [
        str(Path("office_loop", "recording_2020-03-24_17-36-22")),
        str(Path("office_loop", "recording_2020-03-24_17-45-31")),
        str(Path("office_loop", "recording_2020-04-07_10-20-32")),
        str(Path("office_loop", "recording_2020-06-12_10-10-57")),
        str(Path("office_loop", "recording_2021-01-07_12-04-03")),
        str(Path("office_loop", "recording_2021-02-25_13-51-57")),
        str(Path("neighborhood", "recording_2020-03-26_13-32-55")),
        str(Path("neighborhood", "recording_2020-10-07_14-47-51")),
        str(Path("neighborhood", "recording_2020-10-07_14-53-52")),
        str(Path("neighborhood", "recording_2020-12-22_11-54-24")),
        str(Path("neighborhood", "recording_2021-02-25_13-25-15")),
        str(Path("neighborhood", "recording_2021-05-10_18-02-12")),
        str(Path("neighborhood", "recording_2021-05-10_18-32-32")),
        str(Path("business_park", "recording_2020-10-08_09-30-57")),
        str(Path("business_park", "recording_2021-01-07_13-12-23")),
        str(Path("business_park", "recording_2021-02-25_14-16-43")),
        str(Path("countryside", "recording_2020-04-07_11-33-45")),
        str(Path("countryside", "recording_2020-06-12_11-26-43")),
        str(Path("countryside", "recording_2020-10-08_09-57-28")),
        str(Path("countryside", "recording_2021-01-07_13-30-07")),
    ]

    test_sequences = [
        str(Path("city_loop", "recording_2020-12-22_11-33-15")),
        str(Path("city_loop", "recording_2021-01-07_14-36-17")),
        str(Path("city_loop", "recording_2021-02-25_11-09-49")),
        str(Path("old_town", "recording_2020-10-08_11-53-41")),
        str(Path("old_town", "recording_2021-01-07_10-49-45")),
        str(Path("old_town", "recording_2021-02-25_12-34-08")),
        str(Path("old_town", "recording_2021-05-10_21-32-00")),
        str(Path("parking_garage", "recording_2020-12-22_12-04-35")),
        str(Path("parking_garage", "recording_2021-02-25_13-39-06")),
        str(Path("parking_garage", "recording_2021-05-10_19-15-19")),
    ]

