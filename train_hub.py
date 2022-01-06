import hub
import os
import argparse
from typing import NoReturn

import torch
from torch.optim import SGD, lr_scheduler
from torchvision.transforms import functional as F

from model import get_fasterrcnn_resnet50_fpn
from engine import train_one_epoch, evaluate

# ### Global Variables ###
# ## Model ##
TRAINABLE_BACKBONE_LAYERS = 3
# ## Data Fetching ##
BATCH_SIZE = 2
NUM_WORKERS = 2
# ## Optimization ##
LEARNING_RATE = 0.005 * BATCH_SIZE / 4  # Apply linear scaling rule
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
STEP_SIZE = 3
GAMMA = 0.1
# ## Training ##
NUM_EPOCHS = 10
RATIO_TRAINING_SPLIT = 0.8
# ## Logging ##
PRINT_FREQUENCY = 10


def transform_hub_item(hub_item):
    """Transforms a sample from Hub into a model input

    Args:
        hub_item ([type]): expexts "images", "boxes" and "labels" tensors

    Returns:
        [type]: [description]
    """
    image = F.to_tensor(hub_item.get("images"))
    boxes = torch.tensor(hub_item.get("boxes"))
    # Convert hub boxes [x,y,w,h] to [x,y,X,Y]
    boxes[:,2] = boxes[:,0] + boxes[:,2]
    boxes[:,3] = boxes[:,1] + boxes[:,3]
    return (
        image,
        {
            "labels": torch.tensor(hub_item.get("labels")),
            "boxes": boxes,
        },
    )


def collate_fn(batch):
    return tuple(zip(*batch))


def train(training_dataset_path: str, validation_dataset_path: str) -> NoReturn:
    """Train an object detection model on the given dataset. The
    script will store a snapshot of the model after each epoch,
    containing the model's weights and the mapping between the
    model's output and the categories. Those snapshots can directly
    serve as input to the "detect.py" script.

    Args:
        training_dataset_path (str): path to the hub training dataset directory.
        validation_dataset_path (str): path to the hub validation dataset directory.

    Returns:
        NoReturn: [description]
    """
    #### FIXME: The following ugly line is needed on MACOS to avoid https://stackoverflow.com/questions/64772335/pytorch-w-parallelnative-cpp206
    #### This also significantly slows the dataloading process (since it's limited to 1 thread instead of NUM_WORKERS)
    os.environ["OMP_NUM_THREADS"] = "1"
    ####
    training_dataset_hub = hub.load(
        training_dataset_path,
        read_only=True,
    )
    validation_dataset_hub = hub.load(
        validation_dataset_path,
        read_only=True,
    )
    model_output_path = os.path.join(
        "outputs", "models", os.path.basename(training_dataset_path)
    )
    os.makedirs(model_output_path, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Running training on device {}".format(device))
    training_dataloader = training_dataset_hub.pytorch(
        batch_size=BATCH_SIZE, transform=transform_hub_item, collate_fn=collate_fn
    )
    classes = training_dataset_hub.labels.info["class_names"]
    validation_dataloader = validation_dataset_hub.pytorch(
        batch_size=BATCH_SIZE, transform=transform_hub_item, collate_fn=collate_fn
    )
    model = get_fasterrcnn_resnet50_fpn(
        trainable_backbone_layers=TRAINABLE_BACKBONE_LAYERS,
        number_classes=len(classes),
    )
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = SGD(
        params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    # and a learning rate scheduler
    lr_scheduler_training = lr_scheduler.StepLR(
        optimizer, step_size=STEP_SIZE, gamma=GAMMA
    )

    for epoch in range(NUM_EPOCHS):
        train_one_epoch(
            model,
            optimizer,
            training_dataloader,
            device,
            epoch,
            print_freq=PRINT_FREQUENCY,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "categories": classes,
            },
            os.path.join(model_output_path, f"epoch_{epoch}.pth"),
        )
        lr_scheduler_training.step()
        # evaluate on the test dataset
        evaluate(model, validation_dataloader, device=device)


def main():
    parser = argparse.ArgumentParser(description="Train a model on a hub dataset.")
    parser.add_argument(
        "--train-path",
        dest="train_path",
        help="path to your hub training dataset e.g. s3://my-bucket/my-training-dataset",
        required=True,
    )
    parser.add_argument(
        "--val-path",
        dest="val_path",
        help="path to your hub validation dataset e.g. s3://my-bucket/my-validation-dataset",
        required=True,
    )
    args = parser.parse_args()
    train(args.train_path, args.val_path)


if __name__ == "__main__":
    main()
