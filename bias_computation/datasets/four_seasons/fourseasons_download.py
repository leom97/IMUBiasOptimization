import requests
import os
from tqdm import tqdm
import zipfile


# from https://stackoverflow.com/questions/56795227/how-do-i-make-progress-bar-while-downloading-file-in-python
def save_from_url(url, out_file):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_file, 'wb') as f:
            for chunk in tqdm(
                r.iter_content(chunk_size=8192),
                total=int(r.headers.get("content-length", 0))//8192,
                desc=out_file,
                leave=False
            ):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)


data_types = [
    "imu_gnss",
    "point_clouds",
    "reference_poses",
   # "stereo_images_distorted",
   # "stereo_images_undistorted"
]

sequences = {
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/storage/local/mutti/projects/TLIO_dev/Dataset/4s_no_h5py")
    args = parser.parse_args()

    for sequence_location, sequence_dates in tqdm(sequences.items(), desc="Downloading 4seasons dataset"):
        output_dir = os.path.join(args.output_dir, sequence_location)
        os.makedirs(output_dir, exist_ok=True)
        for sequence_date in tqdm(sequence_dates, desc=f"Downloading {len(sequence_dates)} {sequence_location} sequences", leave=False):
            for data_type in tqdm(data_types, desc=sequence_date, leave=False):
                zip_file_name = f"recording_{sequence_date}_{data_type}.zip"
                zip_save_path = os.path.join(output_dir, zip_file_name)
                url = f"https://vision.cs.tum.edu/webshare/g/4seasons-dataset/dataset/" \
                      f"recording_{sequence_date}/{zip_file_name}"
                save_from_url(url, zip_save_path)
                with zipfile.ZipFile(zip_save_path, 'r') as zip_file:
                    zip_file.extractall(path=output_dir)
                os.remove(zip_save_path)
