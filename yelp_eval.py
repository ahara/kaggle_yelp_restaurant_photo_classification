from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime

import csv
import math
import numpy as np
import time
from tensorflow.python.platform import gfile
import tensorflow as tf

import yelp
import yelp_input


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/tmp/yelp_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/yelp_train',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 1,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', len(yelp_input.LBPDict().get_val_photo_ids()),
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, f1_op, summary_op, logits, labels, business):
    """Run Eval once.
    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            f1_sum = 0  # Counts the number of correct predictions.
            step = 0
            while step < num_iter and not coord.should_stop():
                f1 = sess.run(f1_op)
                f1_sum += f1
                step += 1
            # Compute precision @ 1.
            mean_f1 = f1_sum / step
            print('%s: mean F1 @ 1 = %.3f' % (datetime.now(), mean_f1))
            save_results(logits, labels, business, sess)
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Mean F1 @ 1', simple_value=mean_f1)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def save_results(logits, labels, business, sess):
    p = sess.run(logits)
    y = sess.run(labels)
    b = sess.run(business)

    ncol = 19
    nrow = p.shape[0]
    r = np.zeros((nrow, ncol), dtype=float)

    r[:, :9] = y
    r[:, 9:18] = p
    r[:, 18] = b

    with open('preds.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',')
        for i in xrange(nrow):
            writer.writerow(r[i, :])


def evaluate():
    with tf.Graph().as_default():
        # Get images and labels for Yelp validation
        eval_data = FLAGS.eval_data == 'test'
        images, labels, business = yelp.inputs(eval_data=eval_data)
        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = yelp.inference(images)
        # Calculate predictions.
        preds = tf.cast(logits >= 0.5, tf.float32)
        tp = tf.reduce_sum(tf.cast(tf.equal(preds, labels), tf.float32) * preds)
        fp = tf.reduce_sum(tf.cast(tf.equal(preds - labels, 1), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.equal(preds - labels, -1), tf.float32))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_op = 2. * (precision * recall) / tf.maximum(precision + recall, 1.)

        # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            yelp.MOVING_AVERAGE_DECAY)

        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.merge_all_summaries()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.train.SummaryWriter(FLAGS.eval_dir,
                                                graph_def=graph_def)
        while True:
            eval_once(saver, summary_writer, f1_op, summary_op, logits, labels, business)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)


def main(argv=None):  # pylint: disable=unused-argument
    if gfile.Exists(FLAGS.eval_dir):
        gfile.DeleteRecursively(FLAGS.eval_dir)
    gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
