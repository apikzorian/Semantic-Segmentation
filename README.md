# Semantic Segmentation
### Overview
In this project, I trained a fully convolutional neural network (FCN) and used it to classify pixels in an iamge and label those pixels that belong to the road in the image. 

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

## Fully-Connected Network

The main advantage of using fully-convolutional network is that we can preserve spatial information. While the classic fully-connected networks consist of a series of convolutional layers, followed by some fully connected layers, the fully-convolutional network replaces these fully-connected layers with a single 1x1 convolution, followed by a series of transposed convolutional layers. Below is an example of the layout of a fully-convolutional network.


FCNs take advantage of three special techniques:

1. Replacing the fully-connected layers in a network with one by one convolutional layers

2. Up-sampling through the use of transponsed convolutional layers

3. Skip connections


A FCN is usually comprised of two parts: encoder and decoder (as seen below):

![alt tag](https://image.ibb.co/jCvVXQ/FCN.png)

The purpose of the encoder is to extract features from the image, while the decoder is responsible for upscaling the output, so that it ends up the same size as the original image. Another advantage of using an FCN is that since convolutional operations really do not care about the size of the image, FCN can work on an image of any size. In a classic CNN with fully connected layers at the end, the size of the input is always constrained by the size of the fully connected layers. 

### Semantic Segmentation

The idea behind semantic segmentation is assigning meaning to different parts of an object. In this project, we accomplish this on the pixel level, by assigning pixels to a target class (i.e. road, car, person, etc). We've seen well known applications of bounding boxes for indetifying objects in images, such as YOLO and SSD. Bounding boxes are very effective at high frame-per-second, but have their limitations. For example, we can see the limitations when detecting a windy road and trying to draw a box around it. With semantic segmentation, we can derive information about each pixel in an image, rather than partitioning sections of the image into bounding boxes. 



## Implementation




##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.
