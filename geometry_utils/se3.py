from geometry_utils import so3
from scipy.interpolate import interp1d

import numpy


class Se3:
    def __init__(self, R=None, t=None):
        if R is None:
            R = numpy.eye(3)
        if t is None:
            t = numpy.zeros((*R.shape[:-2], 3))

        if R.shape[:-2] != t.shape[:-1]:
            raise ValueError(f"Incompatible dimensions R: {R.shape}, t: {t.shape}")
        self.R = R
        self.t = t

    def __mul__(self, other):
        return Se3(
            numpy.einsum('...jk, ...kl -> ...jl', self.R, other.R),
            numpy.einsum('...jk, ...k -> ...j', self.R, other.t) + self.t
        )

    def __getitem__(self, key):
        return Se3(
            self.R[key],
            self.t[key]
        )

    def inverse(self):
        return Se3(
            numpy.einsum('...jk -> ...kj', self.R),
            -numpy.einsum('...kj, ...k -> ...j', self.R, self.t)
        )

    def __str__(self):
        return f"R: {self.R}\n t: {self.t}"

    def __repr__(self):
        return self.__str__()

    @property
    def q(self):
        return so3.Quaternion(so3.r_to_q(self.R))

    @property
    def T(self):
        T = numpy.zeros((*self.shape, 4, 4))
        T[..., :3, :3] = self.R
        T[..., :3, -1] = self.t
        T[..., -1, -1] = 1
        return T

    def diffs(self, window_size=1):
        return self[:-window_size].inverse() * self[window_size:]

    @property
    def shape(self):
        return self.R.shape[:-2]

    def __len__(self):
        return len(self.R)

    @staticmethod
    def ones(*shape):
        return Se3(
            numpy.tile(numpy.eye(3), (*shape, 1, 1)),
            numpy.zeros((*shape, 3))
        )


def interpolate_se3(poses, timestamps, target_timestamps):
    return Se3(
        so3.interpolate_quaternions(poses.q, timestamps, target_timestamps).R,
        interp1d(timestamps, poses.t, axis=0)(target_timestamps)
    )
