# %% Imports
import numpy as np
import logging

import datasets.four_seasons.fourseasons_dataset as four_seasons
import datasets.euroc.euroc_dataset as euroc
import datasets.NCLT.NCLT_dataset as NCLT
import datasets.oxiod.oxiod_dataset as oxiod

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')


class Data:
    def __init__(self, dataset="four_seasons"):

        self.gt_ts_test = None
        self.gt_ts_train = None
        self.imu_ts_train = None
        self.imu_ts_test = None
        self.rotations_test = None
        self.rotations_train = None
        self.raw_gyros_train = None
        self.raw_gyros_test = None
        self.imu_ts = None
        self.gt_ts = None
        self.raw_gyros = None
        self.rotations = None

        if dataset == "four_seasons":
            self.dataset = four_seasons
            self.dataset_folder = "/storage/user/mutti/atcremers38_local/projects/TLIO_dev/Dataset/4s_no_h5py"
        elif dataset == "euroc":
            self.dataset = euroc
            self.dataset_folder = "/storage/user/mutti/atcremers38_local/projects/TLIO_dev/Dataset/Euroc"
        elif dataset == "NCLT":
            self.dataset = NCLT
            self.dataset_folder = "/storage/user/mutti/atcremers38_local/projects/TLIO_dev/Dataset/NCLT"
        elif dataset == "oxiod":
            self.dataset = oxiod
            self.dataset_folder = "/storage/user/mutti/atcremers38_local/projects/TLIO_dev/Dataset/OxIOD"

    def get_sequence(self, mode="test", number=-1):

        sequences_list = self.dataset.DatasetDefinitions()
        sequence_name = sequences_list.test_sequences[number] if mode == "test" else sequences_list.train_sequences[
            number]

        gt_poses, imu_measurements, gt_timestamps_ns, imu_timestamps_ns = self.dataset.load_sequence(
            self.dataset_folder + "/" + sequence_name,
            subsampling_step=1,
            raw_return=True)

        self.rotations = gt_poses.R
        self.raw_gyros = imu_measurements[:, :3]
        self.gt_ts = gt_timestamps_ns * 1e-9
        self.imu_ts = imu_timestamps_ns * 1e-9

    def subsample_rotations(self, new_dt=5):

        if new_dt == "raw":
            new_dt = 0

        dt_gt = np.diff(self.gt_ts).mean()

        min_dt_ground_truth = new_dt  # seconds

        # ground truth subsampling
        rotations_subs = [self.rotations[0]]
        gt_ts_subs = [self.gt_ts[0]]

        cum_dt = 0
        for i in range(1, len(self.gt_ts) - 1):
            cum_dt += (self.gt_ts[i] - self.gt_ts[i - 1])

            if cum_dt + dt_gt > min_dt_ground_truth:
                cum_dt = 0
                rotations_subs.append(self.rotations[i])
                gt_ts_subs.append(self.gt_ts[i])

        # add last bit in any case
        rotations_subs.append(self.rotations[-1])
        gt_ts_subs.append(self.gt_ts[-1])

        rotations_subs = np.array(rotations_subs)
        gt_ts_subs = np.array(gt_ts_subs)

        return rotations_subs, gt_ts_subs

    def subsample_imu(self, new_freq=2000.):

        if new_freq == "raw":
            new_freq = 2000.

        dt_imu = np.diff(self.imu_ts).mean()

        min_dt_imu = 1 / new_freq
        raw_gyros_subs = [self.raw_gyros[0][:, None]]
        imu_ts_subs = [self.imu_ts[0]]

        cum_dt = 0
        for i in range(1, len(self.imu_ts) - 1):
            cum_dt += (self.imu_ts[i] - self.imu_ts[i - 1])

            if cum_dt + dt_imu > min_dt_imu:
                cum_dt = 0
                raw_gyros_subs.append(self.raw_gyros[i][:, None])
                imu_ts_subs.append(self.imu_ts[i])

        # add last bit in any case
        raw_gyros_subs.append(self.raw_gyros[-1][:, None])
        imu_ts_subs.append(self.imu_ts[-1])

        raw_gyros_subs = np.squeeze(np.array(raw_gyros_subs))
        imu_ts_subs = np.array(imu_ts_subs)

        return raw_gyros_subs, imu_ts_subs

    def build_subsampled_data(self, rot_cgf=None, imu_cfg=None):
        if imu_cfg is None:
            imu_cfg = {"train": "raw", "test": 200}
        if rot_cgf is None:
            rot_cgf = {"train": 5, "test": "raw"}

        self.raw_gyros_test, self.imu_ts_test = self.subsample_imu(new_freq=imu_cfg["test"])
        self.raw_gyros_train, self.imu_ts_train = self.subsample_imu(new_freq=imu_cfg["train"])
        self.rotations_train, self.gt_ts_train = self.subsample_rotations(new_dt=rot_cgf["train"])
        self.rotations_test, self.gt_ts_test = self.subsample_rotations(new_dt=rot_cgf["test"])
