"""
Note, this file is the original one from Linus
"""
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


class Constants:
    gt_filename = 'gt_poses.npz'
    imu_filename = 'imu.npy'
    transformations_filename = 'Transformations.txt'


def interpolate_poses(poses: Se3, pose_timestamps, target_timestamps):
    r_interpolated = so3.interpolate_quaternions(poses.q, pose_timestamps, target_timestamps).R
    t_interpolated = interp1d(pose_timestamps, poses.t, axis=0)(target_timestamps)
    gt_poses = Se3(r_interpolated, t_interpolated)
    gt_velocities = (t_interpolated[1:] - t_interpolated[:-1]) / numpy.diff(target_timestamps * 1e-9)[:, None]
    return gt_poses, gt_velocities


def load_sequence(sequence_dir, subsampling_step, raw_return = False, do_bias_correction=True):
    gt_poses, gt_timestamps = load_gt(os.path.join(sequence_dir, Constants.gt_filename))
    imu_measurements, imu_timestamps = load_imu(sequence_dir, gt_timestamps[0], gt_timestamps[-1])

    if raw_return:
        return gt_poses, imu_measurements, gt_timestamps, imu_timestamps

    if do_bias_correction:
        imu_measurements, biases = imu.get_bias_corrected_imu(
            imu_measurements, imu_timestamps, gt_poses, gt_timestamps
        )
    gt_poses_interpolated, gt_velocities = interpolate_poses(gt_poses, gt_timestamps,
                                                             imu_timestamps[::subsampling_step])
    return Sequence(
        poses=gt_poses_interpolated,
        velocities=gt_velocities,
        imu_measurements=imu_measurements[::subsampling_step],
        timestamps=imu_timestamps[::subsampling_step]
    )


class DatasetDefinitions:
    imu_frequency = 2000.
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


def load_gt(npz_path):
    npz_poses = numpy.load(npz_path)
    return Se3(npz_poses["R"].astype(numpy.float64), npz_poses["t"].astype(numpy.float64)), npz_poses["timestamps"]


def load_imu(sequence_dir, start_timestamp, end_timestamp):
    """Load imu data from file. Ignoring datapoints before start_timestamp."""

    imu_data = numpy.load(os.path.join(sequence_dir, Constants.imu_filename))
    imu_timestamps = imu_data[:, 0].astype(numpy.int64)
    imu_measurements = imu_data[:, 1:]

    transforms = load_transforms(os.path.join(sequence_dir, Constants.transformations_filename))
    imu_measurements = imu.rotate_imu(imu_measurements, transforms["TS_cam_imu"].rotation_matrix.T)

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