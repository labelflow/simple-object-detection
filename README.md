# Simple object detection

<p align="center">
  <img src="public/schema-framework.svg" />
</p>

This work illustrates how to build an object detection proof of concept from image labelling to quantitative evaluation of a model. All you need to get started is a set of raw images that are relevant to the task you want to address. It is supported by a presentation that covered the following topics:

- Labelling of images

- Training of a reference object detection model

- Quantitative and qualitative evaluation

## Stack and inspiration

The inspiration for this work comes from the [torchvision object detection tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) with the minimum requirements to actually use the trained model added on top. The framework is based on PyTorch and most of the utils are just copied from https://github.com/pytorch/vision/tree/main/references/detection. The frameworks' inputs and outputs are all datasets in [COCO](https://cocodataset.org/#format-data) format, which is one of the reference formats for manipulating annotated images.

Data labellisation and visualization are done on [LabelFlow](https://labelflow.ai/) an open-source annotation platform that doesn't need any sign-up and that doesn't store your images.
## 0 - Install the requirements

Make sure that you have python 3.8 installed. It is recommended to create a new virtual environment to avoid interferences with your global libraries.

```
pip install -r requirement.txt
```

Additionally, you should make sure that the version of `torch` you use is compatible with your environment, especially if you wish to use a GPU. Just follow the instructions on [PyTorch](https://pytorch.org/get-started/locally/).

## 1 - Label your images

A viable dataset should have the following properties:

- Big enough: there's between 100 and 1000 labels per class. Keep in mind that the higher this number is, the better the model will get.

- Balanced: there's approximately the same amount of labels per class

- Consistent: the images' distribution reflects the reality of the task you're trying to address

Connect on [LabelFlow](https://labelflow.ai/), upload and label your raw images. Export them to COCO format, making sure that you toggle the options to include the images.

<p align="center">
  <img src="public/labelflow-export.gif" />
</p>

## 2 - Train your model on a custom dataset

```
python train.py --dataset-path <your-coco-format-dataset-directory-path>
```

This script will train a new model for you on a coco dataset that you exported from LabelFlow. One example dataset can be found in `data/sample-coco-dataset`. The model's snapshot weights will be stored after each training epoch in `outputs/models/model_<dataset name>_epoch_<snapshot index>.pth`.

## 3 - Make inferences

```
python detect.py --dataset-path <your-coco-format-dataset-directory-path> --model-path <your-model-snapshot-path>
```

This script runs your model on a coco dataset and generates a coco annotation file containing the inferences in `outputs/inferences/<dataset name>_annotations.json`.
## 4 - Evaluate your model

Additionally to the metrics computed on the validation dataset during training, you can also qualitatively evaluate your prototype by importing the annotation file of the last section in a raw dataset on [LabelFlow](https://labelflow.ai/).

<p align="center">
  <img src="public/labelflow-import.gif" />
</p>

## 5 - Next steps

You can tune the parameters in `train.py` and in `detect.py` to optimize the performance of your prototype. If the results are satisfying, you can add more training data and switch to a more scalable and configurable framework like Detectron2. 
