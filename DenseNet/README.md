# DenseNet

DenseNet got the best paper award in CVPR 2017.

DenseNet was first introduced in the paper **Densely Connected Convolutional Networks** by **Gao Huang**, **Zhuang Liu**, **Kilian Q. Weinberger** and **Laurens van der Maaten**.

This repo includes an overview about DenseNet power, and apply it for CIFAR-10 datasets, you can compare it to ResNet.

Repo structures:
- ``diagram``: the diagram folder of this repo, includes the *loss diagram* and *accuracy diagram*.
- ``model``: model folder contains ``keras`` model files, includes ``densenet40.h5`` (40-layer of DenseNet), ``model_100_stage_1.h5``, ``model_100_final.h5`` (2 stage training 100-layer DenseNet because the training time was too long).
- ``text_file``: correspond to the files in ``model`` folder, stores the ``history`` object (training process) for plotting the diagram.
- ``densenet.py``: source file.
- ``densenet.ipynb``: notebook of this repo.

Did't use data augmentation in this repo, achives about 86% accuracy on 40-layer model, and 88% accuracy on 100-layer model. Similar to ResNet, the deeper version of DenseNet outperform the shallower version.

------
