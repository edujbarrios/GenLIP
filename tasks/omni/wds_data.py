import torch
from torch.utils.data import IterableDataset, DataLoader
from PIL import Image
from io import BytesIO
import json
from datasets import load_dataset
import os
import numpy as np

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from veomni.models import build_processor
import glob

class HFWebDatasetWrapper(IterableDataset):
    def __init__(self, data_path, max_samples=125):
        self.data_path = data_path
        self.max_samples = max_samples

    def __iter__(self):
        # data_files = sorted(glob.glob(os.path.join(self.data_path, "*.tar")))
        # data_files[:125]
        # data_files = [data_files.format(i=i) for i in np.random.choice(len(data_files), self.max_samples, replace=False)]
        # print(data_files)
        tar_list = ["000000.tar", "000001.tar", "000002.tar", "000003.tar"]
        dataset = load_dataset(
            "webdataset",
            data_files={"train": [os.path.join(self.data_path, t) for t in tar_list]},
            split="train",
            streaming=True
        )

        # dataset = load_dataset(
        #     "webdataset",
        #     data_files={"train": data_files},
        #     split="train",
        #     streaming=True
        # )

        dataset = dataset.decode(False)

        for i, sample in enumerate(dataset):
            yield sample
            # print(f"sample {sample}")
            # img = Image.open(BytesIO(sample["jpg"]["bytes"])).convert("RGB")
            # caption = sample["json"]["caption"]

            # yield {"image": img, "caption": caption}

# data_path = "/mnt/bn/zilongdata-us/dataset/recap-datacomp-1b-webdataset/{i:06d}.tar"
data_path = "/mnt/bn/zilongdata-us/dataset/recap-datacomp-1b-webdataset"

dataset = HFWebDatasetWrapper(data_path, max_samples=125)

dataloader = DataLoader(dataset, batch_size=None, num_workers=1, pin_memory=True)
# tokenizer = build_processor("Qwen/Qwen2-VL-7B").tokenizer

max_len = 0
for i, sample in enumerate(dataloader):
    if i % 100 == 0:
        # print(sample.keys())
        print(f"txt {len(sample['txt'])} -- {sample['txt']} \n, json -- {sample['json']}")
    # token_len = len(tokenizer.encode(sample["caption"]))
    # max_len = max(token_len, max_len)
    # if i  % 10 == 0:
        # print(i, np.array(sample["image"]).shape, token_len)
