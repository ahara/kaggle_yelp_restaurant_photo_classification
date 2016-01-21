from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

import tensorflow as tf
from tensorflow.python.platform import gfile

import consts
import yelp_input


FLAGS = tf.app.flags.FLAGS
# Basic model parameters.
tf.app.flags.DEFINE_integer('batch_size', 128, "Number of images to process in a batch.")
tf.app.flags.DEFINE_string('data_dir', consts.PHOTOS_TRAIN, "Path to the Yelp train data directory.")

IMAGE_SIZE = 28
NUM_CLASSES = 9
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 100000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000
# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1       # Initial learning rate.


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    tensor_name = re.sub('tower_[0-9]*/', '', x.op.name)
    tf.histogram_summary(tensor_name + '/activations', x)
    tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
    Returns:
      Variable Tensor
    """
    var = _variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _generate_image_and_label_batch(image, label, business, min_queue_examples):
    """Construct a queued batch of images and labels.
    Args:
      image: 3-D Tensor of [IMAGE_SIZE, IMAGE_SIZE, 3] of type.float32.
      label: 2-D Tensor of type.int32 (multiple-hot-encoding)
      business: 1-D Tensor of type.int32
      min_queue_examples: int32, minimum number of samples to retain
        in the queue that provides of batches of examples.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 2D tensor of [batch_size, classes] size.
      business: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'FLAGS.batch_size' images + labels from the example queue.
    num_preprocess_threads = 2
    images, label_batch, business_batch = tf.train.shuffle_batch(
        [image, label, business],
        batch_size=FLAGS.batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * FLAGS.batch_size,
        min_after_dequeue=min_queue_examples)
    # Display the training images in the visualizer.
    tf.image_summary('images', images)
    return images, tf.reshape(label_batch, [FLAGS.batch_size, NUM_CLASSES]), business_batch


def distorted_inputs():
    """Construct distorted input for Yelp training using the Reader ops.
    Raises:
      ValueError: if no data_dir
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 2D tensor of [batch_size, NUM_CLASSES] size.
    """
    lbp = yelp_input.LBPDict()
    filenames = [os.path.join(consts.PHOTOS_TRAIN, '%d.jpg,%d,%s') %
                 (i, lbp.get_business(i), lbp.get_label_str(i))
                 for i in lbp.get_train_photo_ids()]

    from scipy import misc
    small_images = []

    for f in filenames:
        fname = f.split(',')[0]
        if not gfile.Exists(fname):
            raise ValueError('Failed to find file: ' + fname)
        img = misc.imread(fname)
        if img.shape[0] < IMAGE_SIZE or img.shape[1] < IMAGE_SIZE:
            small_images.append(f)

    print(small_images)
    filenames = [f for f in filenames if f not in small_images]

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    # Read examples from files in the filename queue.
    read_input = yelp_input.read_yelp(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.image.random_crop(reshaped_image, [height, width])
    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,
                                                 max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,
                                               lower=0.2, upper=1.8)
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(distorted_image)
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    print ('Filling queue with %d Yelp images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, read_input.business,
                                           min_queue_examples)


def inputs(eval_data):
    """Construct input for Yelp evaluation using the Reader ops.
    Args:
      eval_data: bool, indicating if one should use the train or eval data set.
    Raises:
      ValueError: if no data_dir
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 2D tensor of [batch_size, NUM_CLASSES] size.
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    if not eval_data:
        lbp = yelp_input.LBPDict()
        filenames = [os.path.join(consts.PHOTOS_TRAIN, '%d.jpg,%d,%s') %
                     (i, lbp.get_business(i), lbp.get_label_str(i))
                     for i in lbp.get_train_photo_ids()]

        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        lbp = yelp_input.LBPDict()
        filenames = [os.path.join(consts.PHOTOS_TRAIN, '%d.jpg,%d,%s') %
                     (i, lbp.get_business(i), lbp.get_label_str(i))
                     for i in lbp.get_val_photo_ids()]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        fname = f.split(',')[0]
        if not gfile.Exists(fname):
            raise ValueError('Failed to find file: ' + fname)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)
    # Read examples from files in the filename queue.
    read_input = yelp_input.read_yelp(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)
    height = IMAGE_SIZE
    width = IMAGE_SIZE
    # Image processing for evaluation.
    # Crop the central [height, width] of the image.
    #resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)
    resized_image = tf.image.random_crop(reshaped_image, [height, width])
    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_whitening(resized_image)
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label, read_input.business,
                                           min_queue_examples)


def inference(images):
    """Build the Yelp model.
    Args:
      images: Images returned from distorted_inputs() or inputs().
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 3, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv1)
    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')
    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[5, 5, 64, 64],
                                             stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        _activation_summary(conv2)
    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        dim = 1
        for d in pool2.get_shape()[1:].as_list():
            dim *= d
        reshape = tf.reshape(pool2, [FLAGS.batch_size, dim])
        weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu_layer(reshape, weights, biases, name=scope.name)
        _activation_summary(local3)
    # local4
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                              stddev=0.04, wd=0.004)
        biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu_layer(local3, weights, biases, name=scope.name)
        _activation_summary(local4)
    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                              stddev=1/192.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.nn.sigmoid(tf.nn.xw_plus_b(local4, weights, biases, name=scope.name))
        _activation_summary(softmax_linear)
    return softmax_linear


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 2-D tensor
              of shape [batch_size, NUM_CLASSES]
    Returns:
      Loss tensor of type float.
    """
    dense_labels = labels
    # Calculate the average cross entropy loss across the batch.
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits, dense_labels, name='cross_entropy_per_example')  # sigmoid layer can be a problem
    #cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    # F1
    preds = logits
    tp = tf.reduce_sum(tf.cast(preds * labels, tf.float32))
    fp = tf.reduce_sum(tf.cast(tf.maximum(preds - labels, 0), tf.float32))
    fn = tf.reduce_sum(tf.cast(tf.maximum(labels - preds, 0), tf.float32))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_op = 2. * (precision * recall) / tf.maximum(precision + recall, 1.)
    cross_entropy_mean = 1 - f1_op
    #cross_entropy_mean = tf.reduce_mean(tf.square(logits - dense_labels))
    tf.add_to_collection('losses', cross_entropy_mean)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    """Add summaries for losses in Yelp model.
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
      total_loss: Total loss from loss().
    Returns:
      loss_averages_op: op for generating moving averages of losses.
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    # Attach a scalar summmary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.scalar_summary(l.op.name +' (raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    return loss_averages_op


def train(total_loss, global_step):
    """Train Yelp model.
    Create an optimizer and apply to all trainable variables. Add moving
    average for all trainable variables.
    Args:
      total_loss: Total loss from loss().
      global_step: Integer Variable counting the number of training steps
        processed.
    Returns:
      train_op: op for training.
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
    print(num_batches_per_epoch, decay_steps)
    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)
    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = _add_loss_summaries(total_loss)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    # Add histograms for gradients.
    for grad, var in grads:
        if grad:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
