import time
import argparse

import hub

from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import functional as F

from dataset import get_coco_dataset
from train_hub import collate_fn, transform_hub_item

# ### Global Variables ###
# ## Data Fetching ##
BATCH_SIZE = 5
NUM_WORKERS = 1
# ## Training ##
RATIO_TRAINING_SPLIT = 1
max_count = 50


def test_local(dataset_path: str):
    train_dataset = get_coco_dataset(dataset_path, train=True)
    dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False,
        collate_fn=collate_fn,
    )
    count = 0
    before = time.time()
    for item in dataloader:
        if count % 10 == 0:
            print("item #", count)
        count += 1
        if count >= max_count:
            break
    after = time.time()
    print(
        """
    Regular local dataset loading method run in {}ms/image
    with config:
        - number workers = {}
        - batch size = {}
    """.format(
            int(1000 * (after - before) / (max_count * BATCH_SIZE)),
            NUM_WORKERS,
            BATCH_SIZE,
        )
    )


def test_hub(dataset_path: str):
    ds = hub.load(
        dataset_path,
        read_only=True,
    )
    dataloader = ds.pytorch(
        batch_size=BATCH_SIZE,
        transform=transform_hub_item,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
    )
    count = 0
    before = time.time()
    for item in dataloader:
        if count % 10 == 0:
            print("item #", count)
        count += 1
        if count >= max_count:
            break
    after = time.time()
    print(
        """
    Hub dataset loading method run in {}ms/image
    with config:
        - number workers = {}
        - batch size = {}
    """.format(
            int(1000 * (after - before) / (max_count * BATCH_SIZE)),
            NUM_WORKERS,
            BATCH_SIZE,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare local dataset loading to remote hub dataset loading")
    parser.add_argument(
        "--local-path",
        dest="local_path",
        help="path to your local coco dataset directory",
        required=True,
    )
    parser.add_argument(
        "--hub-path",
        dest="hub_path",
        help="path to the same dataset as local-path but hosted on hub e.g. s3://my-bucket/my-dataset",
        required=True,
    )
    args = parser.parse_args()
    test_local(args.local_path)
    test_hub(args.hub_path)
