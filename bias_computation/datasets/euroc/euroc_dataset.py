import os
import numpy
import pandas

from geometry_utils.se3 import Se3
from geometry_utils import so3
from dataloading.sequence import Sequence


class DatasetDefinitions:
    imu_frequency = 200.
    train_sequences = [
        'MH_01_easy',
        'MH_03_medium',
        'MH_05_difficult',
        'V1_02_medium',
        'V2_01_easy',
        'V2_03_difficult'
    ]

    test_sequences = [
        'MH_02_easy',
        'MH_04_difficult',
        'V2_02_medium',
        'V1_03_difficult',
        'V1_01_easy',
    ]


def get_poses(data_table):
    translations = numpy.stack([
        data_table[" p_RS_R_x [m]"],
        data_table[" p_RS_R_y [m]"],
        data_table[" p_RS_R_z [m]"],
    ], axis=-1)
    orientations = numpy.stack([
        data_table[" q_RS_w []"],
        data_table[" q_RS_x []"],
        data_table[" q_RS_y []"],
        data_table[" q_RS_z []"],
    ], axis=-1)
    orientations /= numpy.linalg.norm(orientations, axis=-1)[..., None]
    return Se3(so3.q_to_r(orientations), translations)


def get_velocities(data_table):
    return numpy.stack([
        data_table[" v_RS_R_x [m s^-1]"],
        data_table[" v_RS_R_y [m s^-1]"],
        data_table[" v_RS_R_z [m s^-1]"],
    ], axis=-1)


def get_imu(data_table):
    return numpy.stack([
        data_table["w_RS_S_x [rad s^-1]"],
        data_table["w_RS_S_y [rad s^-1]"],
        data_table["w_RS_S_z [rad s^-1]"],
        data_table["a_RS_S_x [m s^-2]"],
        data_table["a_RS_S_y [m s^-2]"],
        data_table["a_RS_S_z [m s^-2]"],
    ], axis=-1)


def get_bias(data_table):
    return numpy.stack([
        data_table[" b_w_RS_S_x [rad s^-1]"],
        data_table[" b_w_RS_S_y [rad s^-1]"],
        data_table[" b_w_RS_S_z [rad s^-1]"],
        data_table[" b_a_RS_S_x [m s^-2]"],
        data_table[" b_a_RS_S_y [m s^-2]"],
        data_table[" b_a_RS_S_z [m s^-2]"],
    ], axis=-1)


def load_sequence(sequence_dir, subsampling_step, do_bias_correction=True, raw_return = False):
    gt_data = pandas.read_csv(os.path.join(sequence_dir, "mav0/state_groundtruth_estimate0/data.csv"))
    imu_data = pandas.read_csv(os.path.join(sequence_dir, "mav0/imu0/data.csv"))

    gt_timestamps = gt_data["#timestamp"].to_numpy()  # for euroc, imu timestamps ~= gt timestamps
    gt_poses = get_poses(gt_data)
    gt_velocities = get_velocities(gt_data)

    imu_data = imu_data.iloc[  # remove imu measurements outside of gt range
               numpy.argmin(numpy.abs(imu_data["#timestamp [ns]"] - gt_timestamps[0])):
               numpy.argmin(numpy.abs(imu_data["#timestamp [ns]"] - gt_timestamps[-1])) + 1
               ].reset_index()
    if not len(imu_data) == len(gt_timestamps):
        raise ValueError(f"Timestamp mismatch")
    imu_measurements = get_imu(imu_data)

    if raw_return:
        return gt_poses, imu_measurements, gt_timestamps, imu_data['#timestamp [ns]'].to_numpy()


    if do_bias_correction:
        imu_measurements -= get_bias(gt_data)

    print(f"timestamp diff: {numpy.abs(imu_data['#timestamp [ns]'] - gt_timestamps).max()}")

    return Sequence(
        poses=gt_poses[::subsampling_step],
        velocities=gt_velocities[::subsampling_step],
        imu_measurements=imu_measurements[::subsampling_step],
        timestamps=gt_timestamps[::subsampling_step]
    )