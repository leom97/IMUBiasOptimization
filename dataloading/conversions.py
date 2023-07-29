import os
import numpy

from libartipy.dataset import Dataset
from libartipy.geometry import CoordinateSystem
from dataloading.fourseasons_dataset import Constants
from tqdm import tqdm

SEQUENCES = sequences = {
    "office_loop": [
        "2020-03-24_17-36-22",
        "2020-03-24_17-45-31",
        "2020-04-07_10-20-32",
        "2020-06-12_10-10-57",
        "2021-01-07_12-04-03",
        "2021-02-25_13-51-57",
    ],
    "neighborhood": [
        "2020-03-26_13-32-55",
        "2020-10-07_14-47-51",
        "2020-10-07_14-53-52",
        "2020-12-22_11-54-24",
        "2021-02-25_13-25-15",
        "2021-05-10_18-02-12",
        # "2021-05-10_18-26-26", reference poses missing
        "2021-05-10_18-32-32",
    ],
    "business_park": [
        "2020-10-08_09-30-57",
        "2021-01-07_13-12-23",
        "2021-02-25_14-16-43",
    ],
    "countryside": [
        "2020-04-07_11-33-45",
        "2020-06-12_11-26-43",
        "2020-10-08_09-57-28",
        "2021-01-07_13-30-07",
        # "2021-01-07_14-03-57", reference poses missing
    ],
    "city_loop": [
        "2020-12-22_11-33-15",
        "2021-01-07_14-36-17",
        "2021-02-25_11-09-49",
    ],
    "old_town": [
        "2020-10-08_11-53-41",
        "2021-01-07_10-49-45",
        "2021-02-25_12-34-08",
        # "2021-05-10_19-51-14", reference poses missing
        "2021-05-10_21-32-00",
    ],
    "parking_garage": [
        "2020-12-22_12-04-35",
        "2021-02-25_13-39-06",
        "2021-05-10_19-15-19",
    ]
}

TRAIN_SEQUENCES = [
    "office_loop",
    "neighborhood",
    "business_park",
    "countryside",
]
TEST_SEQUENCES = [
    "city_loop",
    "old_town",
    "parking_garage"
]


def load_fourseaons_poses(sequence_path, gps=True, coordinate_system=CoordinateSystem.ENU):
    dataset = Dataset(sequence_path)
    dataset.parse_keyframes()
    if gps:
        dataset.set_keyframe_poses_to_gps_poses()

    # load poses and timestamps from dataset
    timestamps = numpy.array(dataset.get_all_kf_timestamps())
    poses = dataset.get_keyframe_poses_in_coordinate_system(coordinate_system)
    poses = [poses[timestamp] for timestamp in timestamps]
    return poses, timestamps, dataset


def load_imu_from_txt(imu_file):
    return numpy.loadtxt(imu_file)


def save_poses_as_npz(out_path, poses, timestamps):
    print('saving to ', out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    numpy.savez(
        out_path,
        timestamps=timestamps,
        R=numpy.array([p.rotation_matrix for p in poses]),
        t=numpy.array([p.translation for p in poses])
    )


def get_sequence_path(sequence_name, sequence_date):
    return os.path.join(sequence_name, f'recording_{sequence_date}')


def generate_dataset(fourseasons_dir, out_dir):
    print(out_dir)
    for sequence_name in tqdm(TRAIN_SEQUENCES + TEST_SEQUENCES):
        for sequence_date in tqdm(SEQUENCES[sequence_name], leave=False, desc=sequence_name):
            sequence_path = get_sequence_path(sequence_name, sequence_date)
            source_path = os.path.join(fourseasons_dir, sequence_path)
            gt_poses, timestamps, dataset = load_fourseaons_poses(source_path, gps=True)
            save_poses_as_npz(os.path.join(out_dir, sequence_path, Constants.gt_filename), gt_poses, timestamps)

            imu_data = numpy.loadtxt(os.path.join(source_path, 'imu.txt'))
            numpy.save(os.path.join(out_dir, sequence_path, Constants.imu_filename), imu_data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fourseasons_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")
    args = parser.parse_args()

    generate_dataset(**args.__dict__)
