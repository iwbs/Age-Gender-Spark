# Adapted from: https://www.tensorflow.org/beta/tutorials/distribute/multi_worker_with_keras

from __future__ import absolute_import, division, print_function, unicode_literals


def main_fun(args, ctx):
  from datetime import datetime
  import math
  import numpy as np
  import tensorflow as tf
  import time

  import inception_resnet_v1
  
  #from tensorflowonspark import TFNode
  #strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
  #tf_feed = TFNode.DataFeed(ctx.mgr, False)

  worker_num = ctx.worker_num
  job_name = ctx.job_name
  task_index = ctx.task_index

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
        age = example[0]
        gender = example[1]
        image = np.frombuffer(example[2], dtype=np.uint8)
        image = np.reshape(image, (160, 160, 3))
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
      
      ds = tf.data.Dataset.from_generator(rdd_generator, (tf.int64, tf.int64, tf.string), (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([160, 160, 3])))
      iterator = ds.make_one_shot_iterator()
      age_labels, gender_labels, images = iterator.get_next()
      images = tf.reverse_v2(images, [-1])
      images = tf.image.per_image_standardization(images)

      train_mode = tf.placeholder(tf.bool)
      age_logits, gender_logits, _ = inception_resnet_v1.inference(images, keep_probability=0.8, phase_train=train_mode, weight_decay=1e-5)

      age_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=age_labels, logits=age_logits)
      age_cross_entropy_mean = tf.reduce_mean(age_cross_entropy)

      gender_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gender_labels, logits=gender_logits)
      gender_cross_entropy_mean = tf.reduce_mean(gender_cross_entropy)

      total_loss = tf.add_n([gender_cross_entropy_mean, age_cross_entropy_mean] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

      age_ = tf.cast(tf.constant([i for i in range(0, 101)]), tf.float32)
      age = tf.reduce_sum(tf.multiply(tf.nn.softmax(age_logits), age_), axis=1)
      abs_loss = tf.losses.absolute_difference(age_labels, age)

      gender_acc = tf.reduce_mean(tf.cast(tf.nn.in_top_k(gender_logits, gender_labels, 1), tf.float32))

      tf.summary.scalar("age_cross_entropy", age_cross_entropy_mean)
      tf.summary.scalar("gender_cross_entropy", gender_cross_entropy_mean)
      tf.summary.scalar("total loss", total_loss)
      tf.summary.scalar("train_abs_age_error", abs_loss)
      tf.summary.scalar("gender_accuracy", gender_acc)

      # Add to the Graph operations that train the model.
      global_step = tf.Variable(0, name="global_step", trainable=False)
      lr = tf.train.exponential_decay(1e-3, global_step=global_step, decay_steps=3000, decay_rate=0.9, staircase=True)
      optimizer = tf.train.AdamOptimizer(lr)
      tf.summary.scalar("lr", lr)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # update batch normalization layer
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step)
    
      init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())




      # # Variables of the hidden layer
      # hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units],
      #                     stddev=1.0 / IMAGE_PIXELS), name="hid_w")
      # hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")
      # tf.summary.histogram("hidden_weights", hid_w)

      # # Variables of the softmax layer
      # sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10],
      #                    stddev=1.0 / math.sqrt(hidden_units)), name="sm_w")
      # sm_b = tf.Variable(tf.zeros([10]), name="sm_b")
      # tf.summary.histogram("softmax_weights", sm_w)

      # x_img = tf.reshape(x, [-1, IMAGE_PIXELS, IMAGE_PIXELS, 1])
      # tf.summary.image("x_img", x_img)

      # hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
      # hid = tf.nn.relu(hid_lin)

      # y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

      # global_step = tf.train.get_or_create_global_step()

      # loss = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
      # tf.summary.scalar("loss", loss)
      # train_op = tf.train.AdagradOptimizer(0.01).minimize(
      #     loss, global_step=global_step)

      # # Test trained model
      # label = tf.argmax(y_, 1, name="label")
      # prediction = tf.argmax(y, 1, name="prediction")
      # correct_prediction = tf.equal(prediction, label)
      # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
      # tf.summary.scalar("acc", accuracy)

      # saver = tf.train.Saver()
      # summary_op = tf.summary.merge_all()
      # init_op = tf.global_variables_initializer()

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

      # Loop until the session shuts down or feed has no more data
      step = 0
      while not sess.should_stop() and not tf_feed.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.

        if args.mode == "train":
          sess.run(init_op)
          
          merged = tf.summary.merge_all()
          train_writer = tf.summary.FileWriter(log_dir, sess.graph)
          
          new_saver = tf.train.Saver(max_to_keep=100)
          ckpt = tf.train.get_checkpoint_state(model_path)
          if ckpt and ckpt.model_checkpoint_path:
            new_saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore and continue training!")
          else:
            pass


        
          # _, summary, step = sess.run([train_op, summary_op, global_step])
          # if (step % 100 == 0) and (not sess.should_stop()):
          #   print("{} step: {} accuracy: {}".format(datetime.now().isoformat(), step, sess.run(accuracy)))
          # if task_index == 0:
          #   summary_writer.add_summary(summary, step)
        else:  # args.mode == "inference"
          pass
          # labels, preds, acc = sess.run([label, prediction, accuracy])
          # results = ["{} Label: {}, Prediction: {}".format(datetime.now().isoformat(), l, p) for l, p in zip(labels, preds)]
          # tf_feed.batch_results(results)
          # print("acc: {}".format(acc))

    print("{} stopping MonitoredTrainingSession".format(datetime.now().isoformat()))

    if sess.should_stop() or step >= args.steps:
      tf_feed.terminate()

    # WORKAROUND FOR https://github.com/tensorflow/tensorflow/issues/21745
    # wait for all other nodes to complete (via done files)
    done_dir = "{}/{}/done".format(ctx.absolute_path(args.model), args.mode)
    print("Writing done file to: {}".format(done_dir))
    tf.gfile.MakeDirs(done_dir)
    with tf.gfile.GFile("{}/{}".format(done_dir, ctx.task_index), 'w') as done_file:
      done_file.write("done")

    for i in range(60):
      if len(tf.gfile.ListDirectory(done_dir)) < len(ctx.cluster_spec['worker']):
        print("{} Waiting for other nodes {}".format(datetime.now().isoformat(), i))
        time.sleep(1)
      else:
        print("{} All nodes done".format(datetime.now().isoformat()))
        break