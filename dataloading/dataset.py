from dataclasses import dataclass
import numpy
from functools import partial
from torch.utils.data import Dataset
from tqdm import tqdm
from multiprocessing import Pool
import os

from dataloading import fourseasons_dataset, euroc_dataset, tumvi_dataset
from dataloading.sequence import Sequence
from geometry_utils import so3
from geometry_utils import imu


@dataclass
class DataConfig:
    # TODO: change these to frequencies
    imu_frequency: float
    window_size: int = 16384  # 2**14  # 1s
    prepend_size: int = 0  # number of measurements to prepend to feature vector
    network_frequency: int = 200  # 2000Hz / 5 = 400 Hz
    sample_step: int = 1600  # 2000 Hz / 5 / 20 = 20 Hz
    global_translation: bool = False
    augment_heading: bool = False
    augment_bias: float = True
    augment_orientation: str = None
    augment_gravity_direction: bool = False
    do_bias_correction: bool = False
    dataset_type: str = "fourseasons"

    def __post_init__(self):
        self.network_step = int(numpy.round(self.imu_frequency / self.network_frequency))


class Phase:
    train = "train"
    val = "val"
    eval = "eval"


class SequenceDataset(Dataset):
    def __init__(self, sequence_dirs, config: DataConfig, base_dir='', phase=Phase.val):
        self.sequence_dirs = sequence_dirs
        self.config = config
        self.base_dir = base_dir
        self.phase = phase

        self.sequences = self.load_sequences()
        self.features, self.targets, self.n_measurements = [], [], []
        for sequence in self.sequences:
            features, targets = self.get_training_data(sequence)
            self.features.append(features)
            self.targets.append(targets)
            self.n_measurements.append(len(features))
        self.sample_indices = self.get_sample_indices()

    def set_phase(self, phase):
        self.phase = phase
        self.sample_indices = self.get_sample_indices()

    def get_sample_indices(self):
        sample_indices = []
        for sequence_index in range(len(self.sequence_dirs)):
            if len(self.features[sequence_index]) < self.config.prepend_size + self.config.window_size:
                print(f"sequence {self.sequence_dirs[sequence_index]} has too few samples, skipping")
            elif self.phase == Phase.train or self.phase == Phase.val:
                sequence_sample_indices = numpy.array([
                    [sequence_index, frame_index, frame_index + self.config.window_size]
                    for frame_index in range(self.config.prepend_size,
                                             # self.config.prepend_size + 32,
                                             # 1)
                                             self.n_measurements[sequence_index] - self.config.window_size - 1,
                                             self.config.sample_step)
                ])
                sample_indices.append(sequence_sample_indices)
            elif self.phase == Phase.eval:
                sample_indices.append(
                    [sequence_index, self.config.prepend_size, self.n_measurements[sequence_index] - 1])
        return numpy.vstack(sample_indices)

    def get_training_data(self, sequence: Sequence):
        if self.config.global_translation:
            imu_measurements = imu.rotate_imu(sequence.imu_measurements, sequence.poses.R)
            gt_displacements = sequence.poses.t[self.config.window_size:] - sequence.poses.t[:-self.config.window_size]
        else:
            imu_measurements = sequence.imu_measurements
            gt_displacements = sequence.poses.diffs(self.config.window_size).t
        return imu_measurements, gt_displacements

    def load_sequence(self, *args, **kwargs):
        if self.config.dataset_type == "fourseasons":
            return fourseasons_dataset.load_sequence(*args, **kwargs)
        elif self.config.dataset_type == "euroc":
            return euroc_dataset.load_sequence(*args, **kwargs)
        elif self.config.dataset_type == "tumvi":
            return tumvi_dataset.load_sequence(*args, **kwargs)
        else:
            raise ValueError(f"Dataset type {self.config.dataset_type} not available")

    def load_sequences(self):
        sequence_data = [
            self.load_sequence(
                os.path.join(self.base_dir, sequence_dir),
                subsampling_step=self.config.network_step,
                do_bias_correction=self.config.do_bias_correction
            )
            for sequence_dir in self.sequence_dirs
        ]
        return sequence_data

    def get_dict(self):
        return {
            "sequence_dirs": self.sequence_dirs,
            "base_dir": self.base_dir,
            "config": self.config.__dict__,
        }

    def __len__(self):
        return len(self.sample_indices)

    @staticmethod
    def augment_heading(features, targets):
        angle = numpy.random.random() * 2 * numpy.pi
        cos = numpy.cos(angle)
        sin = numpy.sin(angle)
        R = numpy.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1]
        ])
        features_rotated = imu.rotate_imu(features, R)
        targets_rotated = numpy.einsum('...ij, ...j -> ...i', R, targets)
        return features_rotated, targets_rotated

    @staticmethod
    def augment_bias(features, range_gyro=1e-3, range_accelerometer=1e-3):
        random_bias = numpy.array([
            (numpy.random.random() - 0.5) * range_gyro / 0.5,
            (numpy.random.random() - 0.5) * range_gyro / 0.5,
            (numpy.random.random() - 0.5) * range_gyro / 0.5,
            (numpy.random.random() - 0.5) * range_accelerometer / 0.5,
            (numpy.random.random() - 0.5) * range_accelerometer / 0.5,
            (numpy.random.random() - 0.5) * range_accelerometer / 0.5,
        ])
        return features.copy() + random_bias

    @staticmethod
    def augment_orientation(features, targets, axis):
        if axis == "all":
            r = so3.uniform_random_so3(())
        else:
            angle = numpy.random.random() * numpy.pi * 2
            r = so3.axis_rotation(angle, axis)
        return imu.rotate_imu(features, r), numpy.einsum('...ij,...j->...i', r, targets)

    @staticmethod
    def augment_gravity(features, angle_range=5.):
        angle_rand = numpy.random.random() * numpy.pi * 2
        vec_rand = numpy.array([numpy.cos(angle_rand), numpy.sin(angle_rand), 0])
        theta_rand = (
                numpy.random.random() * numpy.pi * angle_range / 180.0
        )
        rotvec = theta_rand * vec_rand
        r = so3.q_to_r(so3.rotvec_to_q(rotvec))
        return imu.rotate_imu(features, r)

    @staticmethod
    def augment_gravity_local(features, global_rotations, angle_range=5.):
        angle_rand = numpy.random.random() * numpy.pi * 2
        vec_rand = numpy.array([numpy.cos(angle_rand), numpy.sin(angle_rand), 0])
        theta_rand = (
                numpy.random.random() * numpy.pi * angle_range / 180.0
        )
        rotvec = theta_rand * vec_rand
        r = so3.q_to_r(so3.rotvec_to_q(rotvec))
        features_aug = features.copy()
        g_aug = numpy.einsum('nji,jk,k -> ni', global_rotations, r - numpy.eye(3),
                             numpy.array([0, 0, 9.8]))  # R^TR(angle)g - R^Tg
        features_aug[:, 3:] += g_aug
        return features_aug

    @staticmethod
    def add_noise(features, gyro_noise_std=8e-5, accelerometer_noise_std=1e-3):
        noise = numpy.random.randn(*features.shape)
        features_out = features.copy()
        features_out[..., :3] += noise[..., :3] * gyro_noise_std
        features_out[..., 3:] += noise[..., 3:] * accelerometer_noise_std
        return features_out

    def get_mean_stddev(self):
        all_measurements = numpy.vstack([sequence.imu_measurements for sequence in self.sequences])
        return numpy.mean(all_measurements, axis=0), numpy.std(all_measurements, axis=0)

    def __getitem__(self, item):
        raise NotImplementedError("Can't use base dataset class for training.")


class ImuToTranslationDataset(SequenceDataset):
    def __getitem__(self, item):
        sequence_index, start_frame, end_frame = self.sample_indices[item]
        features = self.features[sequence_index][start_frame - self.config.prepend_size:
                                                 end_frame]
        targets = self.targets[sequence_index][start_frame]
        init_velocities = self.sequences[sequence_index].velocities[start_frame]  # velocity at start the window
        init_rotations = self.sequences[sequence_index].poses.R[start_frame]

        sequence = self.sequences[sequence_index]
        if self.config.augment_orientation is not None:
            features, targets = self.augment_orientation(features, targets, self.config.augment_orientation)
        if self.config.augment_heading:
            features, targets = self.augment_heading(features, targets)
        if self.config.augment_bias:
            features = self.augment_bias(features)
        if self.config.augment_gravity_direction:
            if self.config.global_translation:
                features = self.augment_gravity(features)
            else:
                global_rotations = sequence.poses[start_frame:end_frame].R
                features = self.augment_gravity_local(features, global_rotations)

        return features.T.astype(numpy.float32), targets, init_velocities, init_rotations, sequence_index, start_frame


class GyroDenoisingDataset(SequenceDataset):
    def __getitem__(self, item):
        sequence_index, start_frame, end_frame = self.sample_indices[item]
        sequence = self.sequences[sequence_index]
        features = sequence.imu_measurements[start_frame - self.config.prepend_size:end_frame]
        if self.phase == Phase.train and self.config.augment_bias != 0.0:
            features = self.augment_bias(features,
                                         range_gyro=self.config.augment_bias,
                                         range_accelerometer=self.config.augment_bias)
            features = self.add_noise(features)

        target_rotations = sequence.poses.R[
                           start_frame:end_frame + 1]  # need 1 more absolute pose than relative measurements due to fencepost
        target_translations = sequence.poses.t[start_frame:end_frame + 1]
        target_velocities = sequence.velocities[start_frame:end_frame]
        gt_valid_mask = sequence.poses_valid_mask[start_frame:end_frame]
        return features, target_rotations, target_translations, target_velocities, gt_valid_mask

    def get_empty(self):
        return [numpy.zeros(features.shape) for features in self.features]
