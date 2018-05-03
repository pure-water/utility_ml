#!  /usr/bin/env python
# -*- coding: utf-8 -*-

#__author__      = "Barack Obama"
#__copyright__   = "Copyright 2009, Planet Earth"

import tensorflow as tf
import keras.backend as K
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda



run_meta = tf.RunMetadata()
with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)

    net = MobileNet(input_tensor=tf.placeholder('float32', shape=(1,224,224,3)),include_top=False)

    output = Conv2D(30, 
                        (1,1), strides=(1,1), 
                        padding='same', 
                        name='DetectionLayer'
                    ) 
    output = Reshape((13,13,30, 5,6))(output)


    # Profile
    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
