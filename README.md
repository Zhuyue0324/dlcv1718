# DLCV WS 2018 student project: Stixel Deep Convolutional Network for on-road object detection

a basic network reusing a ResNet 34 / 50 architecture and pretrained weights on ImageNet, with added upsampling to perform binary segmentation between classes Road and Obstacles in a vertically sliced picture, and further functions to interpret it as a distance to an object.

## Binary segmenter on vertically sliced picture

Everything can be run from the IPYNB, provided you have downloaded the leftimg8bit and gtfine datasets from CityScapes, and changed the value of ```ROOT```. You also need to either delete all images that are not labelIds in the ground truth folders, or copy them to other folders (the ```_lido``` in my case) and point towards them.

You may also run ```python3 run.py ... | log.txt```, which can train a network with given parameters, eventually starting from saved weights, but won't visualize or even evaluate anything (its only output is the Cross-Entropy loss of the segmentation task)
