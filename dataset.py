import os
from transforms import ToTensor, Compose, RandomHorizontalFlip
from coco_utils import get_coco, CocoDetection


def get_coco_dataset(dataset_path: str, train=False) -> CocoDetection:
    """TODO: make documentation"""
    annotation_file_path = os.path.join(dataset_path, "annotations.json")
    transforms = []
    transforms.append(ToTensor())
    if train:
        transforms.append(RandomHorizontalFlip(0.5))
    transforms_composed = Compose(transforms)
    coco_detection_dataset = get_coco(
        os.path.join(dataset_path, "images"), annotation_file_path, transforms_composed
    )
    return coco_detection_dataset


def get_number_of_classes(coco_detection_dataset: CocoDetection) -> int:
    dataset = coco_detection_dataset.coco.dataset
    num_classes = len(dataset["categories"]) + 1
    return num_classes

def get_model_categories_metadata(coco_detection_dataset: CocoDetection):
    return list(
        map(
            lambda category: {
                **category,
                "name": "Inference<{}>".format(category["name"]),
            },
            coco_detection_dataset.coco.dataset["categories"],
        )
    )