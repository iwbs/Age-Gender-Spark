# Adapted from: https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_keras

from __future__ import absolute_import, division, print_function, unicode_literals


def main_fun(args, ctx):
  from datetime import datetime
  import math
  import numpy
  import tensorflow as tf
  import time

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

  # Parameters
  IMAGE_PIXELS = 64
  hidden_units = 128

  # Get TF cluster and server instances
  cluster, server = ctx.start_cluster_server(1, args.rdma)

  # Create generator for Spark data feed
  tf_feed = ctx.get_data_feed(args.mode == 'train')

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

  if job_name == "ps":
    server.join()
  elif job_name == "worker":
    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % task_index,
      cluster=cluster)):
      
      ds = tf.data.Dataset.from_generator(rdd_generator, (tf.int64, tf.int64, tf.string), (tf.TensorShape([64, 64, 3]), tf.TensorShape([])))
      ds = ds.batch(args.batch_size)

      saver = tf.train.Saver()
      summary_op = tf.summary.merge_all()
      init_op = tf.global_variables_initializer()

    # Create a "supervisor", which oversees the training process and stores model state into HDFS
    logdir = ctx.absolute_path(args.model)
    print("tensorflow model path: {0}".format(logdir))
    summary_writer = tf.summary.FileWriter("tensorboard_%d" % worker_num, graph=tf.get_default_graph())

    hooks = [tf.train.StopAtStepHook(last_step=args.steps)] if args.mode == "train" else []
    with tf.train.MonitoredTrainingSession(master=server.target,
                                           is_chief=(task_index == 0),
                                           scaffold=tf.train.Scaffold(init_op=init_op, summary_op=summary_op, saver=saver),
                                           checkpoint_dir=logdir,
                                           hooks=hooks) as sess:
      print("{} session ready".format(datetime.now().isoformat()))
      step = 0

    # if sess.should_stop() or step >= args.steps:
    tf_feed.terminate()