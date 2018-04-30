#！ /usr/bin/env python
# -*- coding: utf-8 -*-

#__author__      = "Barack Obama"
#__copyright__   = "Copyright 2009, Planet Earth"

import tensorflow as tf
import keras.backend as K
from keras.applications.mobilenet import MobileNet

run_meta = tf.RunMetadata()
with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)
    net = MobileNet( input_tensor=tf.placeholder('float32', shape=(1,32,32,3)),include_top=False)

    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))