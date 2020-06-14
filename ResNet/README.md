# ResNet

Resnet currently is one of the most-used network architecture in Computer Vision.

Resnet was first introduced in paper **Deep Residual Learning for Image Recognition** by **Kaiming He**, **Xiangyu Zhang**, **Shaoqing Ren**, **Jian Sun**, 2016.

This repo includes an overview of Resnet power and apply it to CIFAR-10 datasets.

Repo structure:

- ``resnet.py``: source code of Resnet.
- ``resnet.ipynb``: train resnet for CIFAR-10 datasets, but when i download this notebook from Colab, it lacks output in some cells.
- ``model_20.h5``, ``model_38.h5``: ``keras`` model files.
- ``accuracy_diagram.png``, ``loss_diagram.png``: loss and accuracy diagram for both Resnet-20 and Resnet-38.

In this repo, i don't use data augmentation like the original paper, and achives around 85% accuracy on test set (still overfitting).

Although i increase the depth of the network, Resnet-38 still slightly outperformed the 20-layer version.

________