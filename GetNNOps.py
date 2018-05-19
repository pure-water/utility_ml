#!  /usr/bin/env python
# -*- coding: utf-8 -*-

#__author__      = "Yao Gang"

import tensorflow as tf
import keras.backend as K
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda


MOBILENET_BACKEND_PATH  = "mobilenet_backend.h5"   # should be hosted on a server


run_meta = tf.RunMetadata()
input_size = 224


input_image = Input(shape=(input_size, input_size, 3))
mobilenet.load_weights(MOBILENET_BACKEND_PATH)

with tf.Session(graph=tf.Graph()) as sess:


    #mobilenet = MobileNet(input_tensor=tf.placeholder('float32', shape=(1,input_size,input_size,3)),include_top=False,input_shape=(input_size,input_size,3))
    mobilenet = MobileNet(include_top=False,input_shape=(input_size,input_size,3))

    #sess.run(mobilenet) 
    #net_out = mobilenet(input_image)


    #output = Conv2D(30, 
    #                     (1,1), strides=(1,1), 
    #                     padding='same', 
    #                     name='DetectionLayer'
    #                 )(net_out)
    #output = Reshape((13,13,30, 5,6))(output)


    # Profile
    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
