# Adapted from: https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_keras

from __future__ import absolute_import, division, print_function, unicode_literals


def main_fun(args, ctx):
  import numpy as np
  import tensorflow as tf
  from tensorflowonspark import TFNode

  strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

  tf_feed = TFNode.DataFeed(ctx.mgr, False)

  def rdd_generator():
    while not tf_feed.should_stop():
      batch = tf_feed.next_batch(1)
      if len(batch) > 0:
        example = batch[0]
        # age = np.array(example[0]).astype(np.int64)
        # age = np.reshape(age, (0,))
        # gender = np.array(example[1]).astype(np.int64)
        # gender = np.reshape(gender, (1,))
        # image = np.array(example[2]).astype(np.float32) / 255.0
        # image = np.reshape(image, (64, 64, 3))
        age = example[0]
        gender = example[1]
        image = example[2]
        yield (age, gender, image)
      else:
        return

  ds = tf.data.Dataset.from_generator(rdd_generator, (tf.int64, tf.int64, tf.string), (tf.TensorShape([64, 64, 3]), tf.TensorShape([])))
  ds = ds.batch(args.batch_size)

  tf_feed.terminate()