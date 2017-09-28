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

## Concepts

### Fully-Connected Network

The main advantage of using fully-convolutional network is that we can preserve spatial information. While the classic fully-connected networks consist of a series of convolutional layers, followed by some fully connected layers, the fully-convolutional network replaces these fully-connected layers with a single 1x1 convolution, followed by a series of transposed convolutional layers. Below is an example of the layout of a fully-convolutional network.


FCNs take advantage of three special techniques:

1. Replacing the fully-connected layers in a network with one by one convolutional layers

2. Up-sampling through the use of transponsed convolutional layers

3. Skip connections


A FCN is usually comprised of two parts: encoder and decoder (as seen below):

![alt tag](https://image.ibb.co/jCvVXQ/FCN.png)

The purpose of the encoder is to extract features from the image, while the decoder is responsible for upscaling the output, so that it ends up the same size as the original image. Another advantage of using an FCN is that since convolutional operations really do not care about the size of the image, FCN can work on an image of any size. In a classic CNN with fully connected layers at the end, the size of the input is always constrained by the size of the fully connected layers. 

One drawback of using convolutions or encoding in general is that we get "tunnel vision", where we look very closely at some features and lose the bigger picture in the end. Information gets lost when moving through layers because of this narrow scope. Skip connections provide a way of retaining the information easily. By connecting the output of one layer to a non-adjacent layer, skip connections allow the network to ustilize information from multiple resolutions that may have been optimized away. This results in the network being able to make more precise segmentation decisions. 

![alt_tag](https://image.ibb.co/mfxcCQ/skipconnections.png)

### Semantic Segmentation

The idea behind semantic segmentation is assigning meaning to different parts of an object. In this project, we accomplish this on the pixel level, by assigning pixels to a target class (i.e. road, car, person, etc). We've seen well known applications of bounding boxes for indetifying objects in images, such as YOLO and SSD. Bounding boxes are very effective at high frame-per-second, but have their limitations. For example, we can see the limitations when detecting a windy road and trying to draw a box around it. With semantic segmentation, we can derive information about each pixel in an image, rather than partitioning sections of the image into bounding boxes. 

![alt_tag](https://image.ibb.co/c7DjsQ/semanticseg.png)

## Implementation

We used the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) to train and test our model. 

### FCN Components

#### Encoder
We made use of the pretrained VGG model for our encoder and utilizied the approach described in [this paper](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). VGG16 model is pretrained on ImageNet for classification and the fully-connected layers are replaced by 1-by-1 convolutions. In the `load_vgg` function, we extract tensors layers 3, 4, and 7 from the saved vgg model, as well as tensors for the input and keep probability. 

In the layers function, we replace the fully connected layer with a 1x1 convolution (line ?):

`    layer = tf.layers.conv2d(vgg_layer7_out, NUM_CLASSES, 1, padding='same', strides=1)`

#### Decoder
To build the decoder, we upsampled the input to the original image size. We created tensors for the 3rd and 4th pooling layers and created transposed convolutions to upsample the input image:

```
layer = tf.layers.conv2d_transpose(layer, NUM_CLASSES, 4, padding='same', strides=2)

layer = tf.layers.conv2d_transpose(layer, NUM_CLASSES, 4, padding='same', strides=2)
```

#### Skipped Connections
We combined the output of two layers: The current layer and a layer further back in the network (typically this is supposed to be a pooling layer).

In `layers()`, we combined the result of the previous layer with the result of the 4th pooling layers through elementwise additon and followed it with another transposed convolution. This step was repeated for the third pooling layer output


### Training

We used an adams optimizer with a 1E-4 learning rate and ran on 50 epochs with a batch size of 4. Since the goal of using this FCN was to assign each pixel to its appropriate class, we were able to usilitze the cross entropy loss function. To do this, we had to reshape the output tensor to a 2D tensor, where each row represents a pixel and each column represents a class.


## Results

Below is a graph that shows an exponential decay in loss over 50 epochs. Training took roughly 25 minutes on a single Titan X GPU

![alt_tag](https://image.ibb.co/fo0myk/B4_e50_l1e4.png)

The results were better than expected given the parameters we used. Initially I had opted to use 300 epochs but quickly realized that although the results were impressive, runtime was an issue. I resolved this by lowering the batch size from 16 to 4 and lowering the number of epochs. Below are a few samples of my results

![alt_tag](https://image.ibb.co/fg1myk/ss_0_new.png)
![alt_tag](https://image.ibb.co/e6Pwyk/um_000017.png)

![alt_tag](https://image.ibb.co/eRyZjQ/uu_000081_og.png)
![alt_tag](https://image.ibb.co/bwVQr5/uu_000081.png)

![alt_tag](https://image.ibb.co/fdA8vb/rsz_ss_2.png)
![alt_tag](https://image.ibb.co/h5QfPQ/um_000057.png)

![alt_tag](https://image.ibb.co/eJMHMG/rsz_ss_3.png)
![alt_tag](https://image.ibb.co/bwpkr5/um_000095.png)

![alt_tag](https://image.ibb.co/bxpAab/ss_4.png)
![alt_tag](https://image.ibb.co/m8T0PQ/um_000015.png)

![alt_tag](https://image.ibb.co/nbUnMG/ss_5.png)
![alt_tag](https://image.ibb.co/btZUJk/uu_000081_1.png)


## Conclusion
Although bounding box methodologies such as YOLO and SSD are widely used for image detection in the deep-learning space, semantic segmentation can provide equal if not better results for testcases such as detecting a road from images taken in a self-driving car. With semantic segmentation, we can derive information about and paint each pixel in an image, there by eliminating the chance of claisifying nearby objects simply because they were "caught in the box". FCNs come in handy with semantic segmentatino, as they allow us to preserve spatial information and can work on images of any size.


##### Run
Run the following command to run the project:
```
python main.py
```
