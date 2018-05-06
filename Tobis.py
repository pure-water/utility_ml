#! /usr/bin/env python 

import tensorflow as tf
import keras.backend as K
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np
from keras.utils import plot_model

#import keras
#def get_trainable_params(model):
#    params = []
#    for layer in model.layers:
#        params += keras.engine.training.collect_trainable_weights(layer)
#    return params


run_meta = tf.RunMetadata()

input_size= 416
max_box_per_image = 10


with tf.Session(graph=tf.Graph()) as sess:

    K.set_session(sess)

    input_image = Input(shape=(input_size, input_size, 3),name="input1")
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image , 4)) 




    mobilenet  = MobileNet(input_tensor=tf.placeholder('float32',shape=(1,input_size,input_size,3)),include_top=False,input_shape=(input_size,input_size,3),weights=None)(input_image)
    detect  = Conv2D(30,(1,1),strides=(1,1),padding='same',name='DetectoinLayer0')(mobilenet)
    detect  = Reshape((13, 13, 5, 6))(detect)
    detect  = Lambda(lambda args: args[0])([detect, true_boxes])
    
    yolo = Model([input_image,true_boxes],detect)
    plot_model(yolo, to_file='yolo.png')
    print(detect)
    yolo.summary()

    optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    yolo.compile(loss='mean_squared_error', optimizer=optimizer)


    #yolo_params = get_trainable_params(yolo)
    #param_grad = tf.gradients(detect, yolo_params)

    #set up the train data
    #X=np.random.rand(input_size,input_size,3)
    #yolo.fit(x=X,steps_per_epoch=1)
    #model.fit(steps_per_epoch=1)
    #mobilenet.evaluate(x=X)

    #Profiling ... 
    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))
