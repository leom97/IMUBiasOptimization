import os
import json


def save_dict_to_json(dictionary, save_path, exists_ok=False):
    if not exists_ok and os.path.exists(save_path):
        raise ValueError(f"file already exists {save_path}")

    base_dir = os.path.dirname(save_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with open(save_path, 'w+') as out_file:
        json.dump(dictionary, out_file, indent=4)


def load_json(json_path):
    with open(json_path, 'r') as json_file:
        return json.load(json_file)