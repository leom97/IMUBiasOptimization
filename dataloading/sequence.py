from dataclasses import dataclass
import numpy

from geometry_utils.se3 import Se3


@dataclass
class Sequence:
    poses: Se3
    velocities: numpy.ndarray
    imu_measurements: numpy.ndarray
    timestamps: numpy.ndarray
    poses_valid_mask: numpy.array = None

    def __post_init__(self):
        if self.poses_valid_mask is None:
            self.poses_valid_mask = numpy.ones(len(self.poses))

    @property
    def gyro(self):
        return self.imu_measurements[..., :3]

    @property
    def accelerometer(self):
        return self.imu_measurements[..., 3:]
