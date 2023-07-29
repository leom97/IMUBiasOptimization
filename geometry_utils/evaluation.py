import os
import torch
import pandas
import numpy
from numba import jit
from tqdm.notebook import tqdm

from geometry_utils.se3 import Se3
from network import model
from dataloading.utils import load_json


@jit(nopython=True)
def hat(n):
    skew_matrix = numpy.zeros((*n.shape[:-1], 3, 3))
    skew_matrix[..., 2, 1] = n[..., 0]
    skew_matrix[..., 1, 2] = -n[..., 0]
    skew_matrix[..., 0, 2] = n[..., 1]
    skew_matrix[..., 2, 0] = -n[..., 1]
    skew_matrix[..., 1, 0] = n[..., 2]
    skew_matrix[..., 0, 1] = -n[..., 2]
    return skew_matrix


@jit(nopython=True)
def so3_exp(n):
    exp_out = numpy.zeros((*n.shape[:-1], 3, 3))
    for i in numpy.ndindex(n.shape[:-1]):
        theta = numpy.sqrt((n[i]**2).sum())
        exp_out[i] = numpy.eye(3)
        if theta > 0:
            sin = numpy.sin(theta)
            cos = numpy.cos(theta)
            n_hat = hat(n[i] / theta)
            exp_out[i] += sin * n_hat + (1 - cos) * n_hat @ n_hat
    return exp_out


@jit(nopython=True)
def integrate_rotation(R, R0=None):
    out = numpy.zeros((len(R) + 1, 3, 3))
    if R0 is None:
        out[0] = numpy.eye(3)
    else:
        out[0] = R0
    for i in range(1, len(R)):
        out[i] = out[i - 1] @ R[i - 1]
    return out


def integrate_acceleration(global_acceleration, dt, v0):
    global_velocities = numpy.concatenate([
        v0[..., None, :], v0 + numpy.cumsum(global_acceleration[..., :-1, :], axis=-2) * dt
    ], axis=-2)
    translation = numpy.concatenate([
        numpy.zeros(v0.shape)[..., None, :],
        numpy.cumsum(global_velocities * dt + global_acceleration * dt ** 2 / 2, axis=-2)
    ], axis=-2)
    return translation


def get_angle_degrees(R):
    return numpy.arccos(numpy.clip(.5 * (numpy.einsum('...ii->...', R) - 1), -1, 1)) * 180 / numpy.pi


def compute_errors(gt_poses, poses):
    pose_errors = gt_poses.inverse() * poses
    error_dict = {
        'displacement': numpy.linalg.norm(gt_poses.t, axis=-1),
        'angle': get_angle_degrees(gt_poses.R),
        'translation_error': numpy.linalg.norm(pose_errors.t, axis=-1),
        'rotation_error': get_angle_degrees(pose_errors.R),
    }
    return pandas.DataFrame(error_dict)


def compute_segment_errors(gt_poses, poses, segment_starts, segment_ends):
    gt_diffs = gt_poses[segment_starts].inverse() * gt_poses[segment_ends]
    pose_diffs = poses[segment_starts].inverse() * poses[segment_ends]
    return compute_errors(gt_diffs, pose_diffs)


def get_segment_ends(distances, segment_length, segment_start_step):
    segment_starts = numpy.arange(0, len(distances), segment_start_step)
    segment_ends = numpy.zeros(len(segment_starts), dtype=numpy.int)
    for idx, segment_start in enumerate(tqdm(segment_starts, leave=False)):
        segment_ends[idx] = numpy.searchsorted(distances - distances[segment_start], segment_length)

    # remove segments that start within segment_length of the end
    valid_indices, = numpy.where(segment_ends != len(distances))
    return pandas.DataFrame({
        'segment_end': segment_ends[valid_indices],
        'segment_start': segment_starts[valid_indices]
    })


def compute_gyro_segment_errors(gyro_R, gt_poses, segment_lengths, segment_step=160):
    segments_table = pandas.concat([
        pandas.DataFrame({
            'segment_start': [i for i in range(0, len(gt_poses) - w, segment_step)],
            'segment_end': [i + w for i in range(0, len(gt_poses) - w, segment_step)]
        }).assign(segment_length=w)
        for w in segment_lengths if len(gt_poses) > w
    ])
    segments_table = pandas.concat([
        segments_table.reset_index(drop=True),
        compute_segment_errors(gt_poses, Se3(gyro_R), segments_table.segment_start.to_numpy(),
                               segments_table.segment_end.to_numpy())
    ], axis=1)
    return segments_table


def get_gyro(sequence, network=None):
    if network is None:
        return sequence.gyro
    else:
        return network(torch.Tensor(sequence.imu_measurements[None]).detach().numpy())[0]


def get_network(checkpoint_path):
    config_dict = load_json(os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "configs.json"))
    if "init_channels" not in config_dict["training"]:
        config_dict["training"]["init_channels"] = 32
    config = model.TrainingConfig(**{key: value
                                     for key, value in config_dict["training"].items()
                                     if key != "optimizer_kwargs"})

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    network, _, _ = model.get_model("denoising", config)
    network = network.cpu()
    network.eval()
    network.load_state_dict(checkpoint.get("model_state_dict"))

    return network


def get_errors_table(sequences, network=None, bias=None, start_frame=0):
    error_tables = []
    if isinstance(network, str):
        network = get_network(network)

    network_outputs = []
    for sequence_name, sequence in tqdm(sequences):
        if network is None:
            gyro = sequence.gyro[start_frame:] - (bias[None] if bias is not None else 0.)
            gt_poses = sequence.poses[start_frame:]
            global_accelerations = numpy.einsum('...ij,...j->...i', gt_poses.R, sequence.accelerometer[start_frame:]) - numpy.array([0, 0, 9.81])
        else:
            imu_denoised = \
            network(torch.Tensor(sequence.imu_measurements[None]).to(torch.float32)).detach().cpu().numpy()[0].astype(
                numpy.float64)
            if imu_denoised.shape[1] == 6:
                start_frame = network.receptive_field
                gyro = imu_denoised[:, :3]
                gt_poses = sequence.poses[network.receptive_field:]
                global_accelerations = numpy.einsum('...ij,...j->...i', gt_poses.R, imu_denoised[:, 3:]) - numpy.array([0, 0, 9.81])
            else:
                start_frame = 0
                gyro = imu_denoised
                gt_poses = sequence.poses
                global_accelerations = None
        network_outputs.append({"gyro_rotations": so3_exp(gyro * 1 / 200.)[None],
                                "global_accelerations": global_accelerations,
                                "imu_measurements": sequence.imu_measurements,
                                "target_rotations": gt_poses.R,
                                "gyro_denoised": gyro})
        # error_tables.append(
        #     pandas.DataFrame(model.compute_sequence_metrics({"gyro_rotations": gyro_R[None]}, sequences, 1/200.))
        # compute_gyro_segment_errors(gyro_R, gt_poses, [16, 32, 64, 128, 256, 512, 1024, 4096, 16384]).assign(
        #     sequence=sequence_name)
        # )
    return pandas.DataFrame(model.compute_sequence_metrics(network_outputs, [s for _, s in sequences], 1 / 200.,
                                                           start_frame), index=[0]), network_outputs
