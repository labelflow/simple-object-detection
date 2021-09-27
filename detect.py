import os
import json
import copy
import argparse
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data.dataloader import DataLoader
from utils import collate_fn

from dataset import get_coco_dataset, get_number_of_classes

# ### Global Variables ###
DEBUG=False
# ## Model ##
CONFIDENCE_SCORE_THRESHOLD=0.5
# ## Data Fetching ##
BATCH_SIZE = 8
NUM_WORKERS = 4


def detect(dataset_path: str, model_path: str):
    inferences_output_path = os.path.join("outputs", "inferences")
    os.makedirs(inferences_output_path, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    dataset = get_coco_dataset(dataset_path, train=False)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True,
        collate_fn=collate_fn,
    )
    model_metadata = torch.load(model_path)
    state_dict = model_metadata.get("state_dict")
    categories = model_metadata.get("categories")
    model = fasterrcnn_resnet50_fpn(
        pretrained=False, num_classes=len(categories) + 1,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    result_dataset = copy.deepcopy(dataset.coco.dataset)
    result_dataset["annotations"] = []
    result_dataset["categories"] = categories
    id_annotation = 1
    for images, targets in dataloader:
        images = list(img.to(device) for img in images)
        outputs = model(images)
        for target, output in zip(targets, outputs):
            boxes = output["boxes"].to("cpu")
            labels = output["labels"].to("cpu")
            scores = output["scores"].to("cpu")
            for index_detection in range(boxes.shape[0]):
                bbox = boxes[index_detection].tolist()
                category_id = labels[index_detection].tolist()
                score = scores[index_detection].tolist()
                if score > CONFIDENCE_SCORE_THRESHOLD:
                    if DEBUG:
                        print("[ Annotation ] {} with score {}".format(result_dataset["categories"][category_id-1].get("name"), score))
                    width = bbox[0] - bbox[2]
                    height = bbox[3] - bbox[1]
                    result_dataset["annotations"].append(
                        {
                            "id": id_annotation,
                            "image_id": target["image_id"].item(),
                            "category_id": category_id,
                            "segmentation": [
                                [
                                    bbox[0],
                                    bbox[1],
                                    bbox[2],
                                    bbox[1],
                                    bbox[2],
                                    bbox[3],
                                    bbox[0],
                                    bbox[3],
                                    bbox[0],
                                    bbox[1],
                                ]
                            ],
                            "area": width * height,
                            "bbox": [bbox[0], bbox[1], width, height],
                            "iscrowd": 0,
                        }
                    )
                    id_annotation += 1
    print("Created {} annotations.".format(id_annotation))
    json.dump(
        result_dataset,
        open(
            os.path.join(
                inferences_output_path,
                "{}_coco-annotations.json".format(os.path.basename(dataset_path)),
            ),
            "w",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description="Make inference on a coco dataset.")
    parser.add_argument(
        "--dataset-path",
        dest="dataset_path",
        help="path to your coco dataset directory",
        required=True,
    )
    parser.add_argument(
        "--model-path",
        dest="model_path",
        help="path to your model weights",
        required=True,
    )
    args = parser.parse_args()
    detect(args.dataset_path, args.model_path)


if __name__ == "__main__":
    main()
