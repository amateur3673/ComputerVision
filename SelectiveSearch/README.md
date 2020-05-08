# Selective Search Segmentation

This is a simple implementation of ``Selective Search`` algorithm.

You can find the paper [here](http://www.huppelen.nl/publications/selectiveSearchDraft.pdf)

File structure:
- ``Images``: Folder of image input
- ``result`` files: collected from Graph Segmentation for initial components.
- ``img_transform.py``: for image rotating, takes from ``Image_Transform``.
- ``source.py``: source code of this project.
- ``Segmentation_Image``: segmentation image result.

**Bottleneck**:

There are several slow stages in this project. I'm trying to fix it:

- ``initial_merge``: this is a slow stage
- ``set_image`` in ``class`` ``TextureSimilarity``: slows at rotation, can be replaced by ``warpAffine`` in ``opencv``.

The result is not very good, it may need some tricks.

------