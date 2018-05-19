#! /usr/bin/env python 
"yao gang profile code on SqueezeNet at 2018.05.19"

import tensorflow as tf
import keras.backend as K
from keras.applications.mobilenet import MobileNet
from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
import numpy as np
from keras.utils import plot_model

#import tensorflow.contrib.eager as tfe 
#tfe.enable_eager_execution()

input_size= 224 
max_box_per_image = 10

# define some auxiliary variables and the fire module
sq1x1  = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu   = "relu_"

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x     = Activation('relu', name=s_id + relu + sq1x1)(x)

    left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left  = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=3, name=s_id + 'concat')

    return x

g = tf.Graph()
run_meta = tf.RunMetadata()

with g.as_default():


    input_image = Input(tensor=tf.placeholder('float32',shape=(1,input_size,input_size,3)),shape=(input_size, input_size, 3),name="input1")
    true_boxes  = Input(shape=(1, 1, 1, max_box_per_image , 4)) 

    x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)



    squeezenet = x 
    #mobilenet  = MobileNet(input_tensor=tf.placeholder('float32',shape=(1,input_size,input_size,3)),include_top=False,input_shape=(input_size,input_size,3),weights=None)(input_image)
    ##mobilenet  = MobileNet(include_top=False,input_shape=(input_size,input_size,3),weights=None)(input_image)
    print("squeezenet")
    print(squeezenet)
    detect  = Conv2D(30,(1,1),strides=(1,1),padding='same',name='DetectoinLayer0')(squeezenet)
    print(detect)
    detect  = Reshape((13, 13, 5, 6))(detect)
    detect  = Lambda(lambda args: args[0])([detect, true_boxes])
    
    yolo = Model([input_image,true_boxes],detect)

    plot_model(yolo, to_file='yolo_squezenet.png')
    print(detect)
    yolo.summary()

    #Profiling ... 
    #opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    #flops = tf.profiler.profile(g, run_meta=run_meta, cmd='scope', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(g, run_meta=run_meta, cmd='scope', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)


    print("flops_total: {:,} --- params_total {:,}".format(flops.total_float_ops, params.total_parameters))
