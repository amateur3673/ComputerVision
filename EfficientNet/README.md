# EfficientNet

EfficientNet was released in the paper **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**, 2020, by two authors: **Mixing Tan**, **Quoc V.Le** (Google Brain).

This repo includes a brief overview of EfficientNet power by finetuning it for CIFAR-10 and CIFAR-100 datasets from pretrained ImageNet model.

Repo's structure:
- ``model``: model files for training CIFAR-10 datasets, there are 2 stages of the training process, the first stage is training the classifier and the second stage is fine-tuning model. There isn't CIFAR100 model in this repo because Colab is exceeds time limit during training process.
- ``EfficientNet.py``: file used for fine-tuning model.

In this repo, i achives 94.44% accuracy on CIFAR-10 datasets, and reach 80% accuracy on CIFAR-100 datasets

-----