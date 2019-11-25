# Copyright 2017 Yahoo Inc.
# Licensed under the terms of the Apache 2.0 license.
# Please see LICENSE file in the project root for terms.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import mnist

import cv2
import numpy as np
import os

from tensorflowonspark import TFCluster
from tensorflowonspark import dfutil
import dist

if __name__ == "__main__":
  import argparse

  from pyspark.sql import SparkSession
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf

  sc = SparkContext(conf=SparkConf().setAppName("imdb_train"))
  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1
  num_ps = 1
  spark = SparkSession(sc)

  parser = argparse.ArgumentParser()
  parser.add_argument("--num-partitions", help="Number of output partitions", type=int, default=10)
  parser.add_argument("--model", help="HDFS path to save/load model during train/inference", default="imdb_model")
  parser.add_argument("--path", help="HDFS path")
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  parser.add_argument("--epochs", help="number of epochs", type=int, default=100)
  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=64)
  parser.add_argument("--mode", help="train|inference", default="train")
  args = parser.parse_args()
  print("args:", args)

  tfr_rdd = sc.newAPIHadoopFile(args.path, "org.tensorflow.hadoop.io.TFRecordFileInputFormat",
                                keyClass="org.apache.hadoop.io.BytesWritable",
                                valueClass="org.apache.hadoop.io.NullWritable")

  # infer Spark SQL types from tf.Example
  record = tfr_rdd.take(1)[0]
  example = tf.train.Example()
  example.ParseFromString(bytes(record[0]))
  schema = dfutil.infer_schema(example, binary_features=['image_raw'])

  # convert serialized protobuf to tf.Example to Row
  example_rdd = tfr_rdd.mapPartitions(lambda x: dfutil.fromTFExample(x, binary_features=['image_raw']))
  #df = dfutil.loadTFRecords(sc, args.path, binary_features=['image_raw'])
  #df.show()
  
  cluster = TFCluster.run(sc, dist.main_fun, args, args.cluster_size, num_ps, args.tensorboard, TFCluster.InputMode.SPARK)
  #cluster.train(example_rdd, args.epochs)
  cluster.shutdown()