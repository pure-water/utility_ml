#! /usr/bin/env python 

import tensorflow as tf
import keras.backend as K
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np

run_meta = tf.RunMetadata()

input_size= 224


with tf.Session(graph=tf.Graph()) as sess:

    K.set_session(sess)

    input_image = Input(shape=(input_size, input_size, 3))
    mobilenet = MobileNet(input_tensor=tf.placeholder('float32',shape=(1,224,224,3)),include_top=False,input_shape=(input_size,input_size,3))(input_image)

    final = Conv2D(30,(1,1),strides=(1,1),padding='same',name='DetectoinLayer')(mobilenet)

    model = Model(input_image,final)
    model.summary()

    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    

    #sess.run(feature)
    #mobilenet = MobileNet(input_tensor=tf.placeholder('float32', shape=(1,224,224,3)),include_top=False,input_shape=(input_size,input_size,3))
    #mobilenet = MobileNet(include_top=False,input_shape=(input_size,input_size,3))
    #net.get_output_shape()
    #m_feature = mobilenet(input_image)

    #print(mobilenet.outputs)
    #print(mobilenet.layers)



    #learning_rate = 0.1
    #optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    #output = Conv2D(30, 
    #                     (1,1), strides=(1,1), 
    #                     padding='same', 
    #                     name='DetectionLayer'
    #                 )(m_feature)
 

    #mobilenet.compile(loss='mean_squared_error', optimizer=optimizer)


    #X=np.random.rand(input_size,input_size,3)
    #mobilenet.fit(x=X,steps_per_epoch=1)
    #model.fit(steps_per_epoch=1)
    #mobilenet.evaluate(x=X)

    #Profiling ... 
    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
