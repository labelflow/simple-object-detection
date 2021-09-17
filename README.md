# Simple object detection

This work illustrates how to build an object detection proof of concept from image labelling to quantitative evaluation of a model. All you need to get started is a set of raw images that are relevant to your task. It is meant to be as easy and quick as possible to go from an idea - "Can I detect X on an image?" - to the demonstration of a simple yet effective model trained on a few images. The counterpart of the simplicity of this framework is that it is not configurable, which shouldn't have too much impact for a proof of concept.

It is supported by a presentation that covers the following topics:

- Labelling of images

- Training of a reference object detection model

- Quantitative and qualitative evaluation

## Get Started

Make sure that you have python 3.8 installed. It is recommended to create a new virtual environment to avoid interferences with your global libraries.

```
pip install -r requirement.txt
```

## Train your model on a custom dataset

```
python train.py --dataset-path <your-coco-format-dataset-directory-path>
```

This script will train a new model for you on a coco dataset. The model's snapshot weights will be stored in `outputs/models/<dataset name>_<snapshot step>.pth`.

## Evaluate model

```
python evaluate.py --dataset-path <your-coco-format-dataset-directory-path>
```

This script will log a bunch of metrics accounting for the performance of the model.

## Make inference

```
python detect.py --dataset-path <your-coco-format-dataset-directory-path> --model-path <your-model-snapshot-path>
```

This script runs inferences on a coco dataset and generates an coco annotation file in `outputs/inferences/<dataset name>_annotations.json`.