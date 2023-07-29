import numpy


def get_bias_corrected_imu(imu_measurements, imu_timestamps, gps_poses, gps_timestamps):
    biases, _, _ = get_calibration_from_standstill(imu_measurements, imu_timestamps, gps_poses, gps_timestamps)
    imu_measurements_bias_corrected = imu_measurements - biases
    return imu_measurements_bias_corrected, biases


def get_calibration_from_standstill(imu_measurements, imu_timestamps, gps_poses, gps_timestamps):
    # select segments where the car is standing still for >5sec TODO: base this on actual distance
    standstill_indices, = numpy.where(numpy.diff(gps_timestamps) > 5e9)

    # ignore first/last 1s
    standstill_measurements = []
    imu_standstill_indices = []
    for standstill_index in standstill_indices:
        start_index = numpy.argmin(numpy.abs(imu_timestamps - (gps_timestamps[standstill_index] + 1e9)))
        end_index = numpy.argmin(numpy.abs(imu_timestamps - (gps_timestamps[standstill_index + 1] - 1e9)))
        imu_standstill_indices = [*imu_standstill_indices, *list(range(start_index, end_index))]

        gravity_direction = gps_poses[standstill_index].R.T.dot([0, 0, 9.81])
        standstill_measurements.append(
            numpy.hstack([
                imu_measurements[start_index:end_index, :3],
                imu_measurements[start_index:end_index, 3:] - gravity_direction
            ])
        )
    if len(standstill_indices) == 0:
        return numpy.zeros(6), [], []  # not enough data
    else:
        return numpy.vstack(standstill_measurements).mean(axis=0).astype(numpy.float64), standstill_indices, imu_standstill_indices


def rotate_imu(imu_measurements, rotation_matrix):
    return numpy.hstack([
        numpy.einsum('...ij,...j->...i', rotation_matrix, imu_measurements[:, :3]),
        numpy.einsum('...ij,...j->...i', rotation_matrix, imu_measurements[:, 3:])
    ]).astype(numpy.float64)
