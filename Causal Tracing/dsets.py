import json
import typing
from pathlib import Path
import os
import torch
from torch.utils.data import Dataset



class CTDataset(Dataset):
    def __init__(self, data_dir: str, *args, **kwargs):
        if not os.path.exists(data_dir):
            raise Exception(f"{data_dir} does not exist.")
            # torch.hub.download_url_to_file(REMOTE_URL, known_loc)

        with open(data_dir, "r") as f:
            self.data = json.load(f)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]