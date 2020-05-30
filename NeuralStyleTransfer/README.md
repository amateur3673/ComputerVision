# Neural Style Transfer

This is a simple implementation of Neural Style Transfer problem for generating artistic image.

Refer: Tensorflow tutorials

There are still some problems of this repo, especially with the image resolution. In this repo, i train the image of shape (224,224) (the input shape of the VGG19 network), and i found that this is the best input shape. When i use (448,448) and higher shape, i get the problem, that is it's so difficult to train the style loss. I can't figure out the solution.

So all the images are resized to (224,224) to feed to the network. After training, i resize it once again to (448,448) for higher resolution, so these results are fucking worse compared to the paper.