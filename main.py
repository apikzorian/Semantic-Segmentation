import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

EPOCHS = 300
BATCH_SIZE = 16
LEARN_RATE = 1e-4
NUM_CLASSES = 2

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # Load the model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    vgg = tf.get_default_graph()

    input_img = vgg.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = vgg.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = vgg.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = vgg.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = vgg.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_img, keep_prob, layer3, layer4, layer7

tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Creates the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :return: The Tensor for the last layer of output
    """

    # Encoder: Replacing fully connected layer with 1x1 convolution
    layer = tf.layers.conv2d(vgg_layer7_out, NUM_CLASSES, 1, padding='same', strides=1)

    """
    Decoder
    1. Create 1x1 tensors for 3rd and 4th pooling layers in FCN-8
    2. Create transposed convolution to upsample to original image size
    3. Skip connections by combining outputs of two layers
    """

    pool_3 = tf.layers.conv2d(vgg_layer3_out, NUM_CLASSES, 1, padding='same', strides=1)
    pool_4 = tf.layers.conv2d(vgg_layer4_out, NUM_CLASSES, 1, padding='same', strides=1)

    # Upsample 1: To match with pool_3, upsample by 2
    layer = tf.layers.conv2d_transpose(layer, NUM_CLASSES, 4, padding='same', strides=2)

    # Skip connection 1: Combine result of layer with result of 4th pooling layer
    layer = tf.add(layer, pool_4)

    # Upsample 2:To match with pool_3, upsample by 2
    layer = tf.layers.conv2d_transpose(layer, NUM_CLASSES, 4, padding='same', strides=2)

    # Skip connection 2: Combine result of layer with result of 3rd pooling layer
    layer = tf.add(layer, pool_3)

    # Upsample 3: To resize to original image, upsample by 5
    layer = tf.layers.conv2d_transpose(layer, NUM_CLASSES, 16, padding='same', strides=8)

    return layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Generates TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Reshape so that logits and labels are 2-D tensors where row represents a pixel and column represents a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label_2D = tf.reshape(correct_label, (-1, num_classes))

    # Apply cross-entropy for loss function
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=correct_label_2D, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)

    # Use Adam Optimizer for training
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_operation = optimizer.minimize(loss_operation)

    return logits, training_operation, loss_operation


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    batches = 0
    sess.run(tf.global_variables_initializer())

    print("Number of Epochs: ", EPOCHS)
    print("Batch Size: ", BATCH_SIZE)
    print("Learning rate: ", LEARN_RATE)
    print("Being training...")

    loss_list = []
    for i in range(EPOCHS):
        batch_num = 0
        for images, labels in get_batches_fn(BATCH_SIZE):
            if images.shape[0] != BATCH_SIZE:
                if not batches:
                    batches = batch_num
                continue
            batch_num += 1
            _, loss = sess.run([train_op, cross_entropy_loss],
                               feed_dict={learning_rate: LEARN_RATE, correct_label: labels,
                                          keep_prob: 1.0, input_image: images})
            loss_list.append(loss)
            print("Epoch {}, Batch {}, Loss {:.5f}".format((i+1), batch_num, loss))

    return loss_list, batches

tests.test_train_nn(train_nn)


def plot_loss(loss_list, batches):
    # Plot loss over epochs
    total_batches = batches * EPOCHS
    print(total_batches)
    print(batches)
    x = list(range(0, total_batches, batches))
    epoch_names_in = list(range(0, EPOCHS))
    epoch_names = []
    for i in epoch_names_in:
        epoch_names.append(str(i + 1))
    loss_plot = plt.subplot(111)

    loss_plot.set_title('Loss vs Epochs')
    plt.xticks(x, epoch_names, rotation='vertical')
    loss_plot.set_xlabel('Epoch')
    loss_plot.set_ylabel('Loss')
    loss_plot.set_ylim([0, 1.0])
    loss_plot.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    loss_plot.plot(loss_list)

    plt.show()

def run():

    # Initialize values
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    label = tf.placeholder(tf.float32, shape=[BATCH_SIZE, image_shape[0], image_shape[1], NUM_CLASSES])
    learn_rate = tf.placeholder(tf.float32, shape=[])

    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    #helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        last_layer = layers(layer3_out, layer4_out, layer7_out, NUM_CLASSES)
        logits, training_operation, loss_operation = optimize(last_layer, label, learn_rate, NUM_CLASSES)

        loss_list, batches = train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, training_operation, loss_operation, input_image,
                 label, keep_prob, learn_rate)

        plot_loss(loss_list, batches)

        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
