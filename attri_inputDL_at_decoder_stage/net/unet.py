# Simplified unet for fault segmentation
# The original u-net architecture is more complicated than necessary 
# for our task of fault segmentation.
# We significanlty reduce the number of layers and features at each 
# layer to save GPU memory and computation but still preserve high 
# performace in fault segmentation.

import tensorflow as tf
from turtle import shape
import numpy as np 
import os
#import skimage.io as io
#import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

# def unet(pretrained_weights = None,input_size = (128,128,128,1)):
def unet(input_size1 = (128,128,128,1),input_size2 = (128,128,128,19),input_size3 = (128,128,128,19),input_size4 = (128,128,128,19),input_size5 = (128,128,128,19)):

    nf1 = 32
    nf2 = nf1*2
    nf3 = nf2*2
    nf4 = nf3*2
    nf5 = nf4*2
    inputs = Input(input_size1)#new
    gx0 = Input(input_size2)#new 
    gx1 = Input(input_size3)#new 
    gx2 = Input(input_size4)#new
    gx3 = Input(input_size5)#new
    # inputs = Input(input_size)
    conv1 = Conv3D(nf1, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(inputs)
    conv1 = Conv3D(nf1, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,2,2))(conv1)

    conv2 = Conv3D(nf2, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(pool1)
    conv2 = Conv3D(nf2, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Conv3D(nf3, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(pool2)
    conv3 = Conv3D(nf3, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)

    conv4 = Conv3D(nf4, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(pool3)
    conv4 = Conv3D(nf4, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2,2,2))(conv4)

    conv5 = Conv3D(nf5, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(pool4)
    conv5 = Conv3D(nf5, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv5)

    up6 = concatenate([UpSampling3D(size=(2,2,2))(conv5), conv4, gx3], axis=-1)
    conv6 = Conv3D(nf4, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(up6)
    conv6 = Conv3D(nf4, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv6)

    up7 = concatenate([UpSampling3D(size=(2,2,2))(conv6), conv3, gx2], axis=-1)
    conv7 = Conv3D(nf3, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(up7)
    conv7 = Conv3D(nf3, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv7)

    up8 = concatenate([UpSampling3D(size=(2,2,2))(conv7), conv2, gx1], axis=-1)
    conv8 = Conv3D(nf2, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(up8)
    conv8 = Conv3D(nf2, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv8)

    up9 = concatenate([UpSampling3D(size=(2,2,2))(conv8), conv1, gx0], axis=-1)
    conv9 = Conv3D(nf1, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(up9)
    conv9 = Conv3D(nf1, (3,3,3), activation='relu',  dilation_rate=1, padding='same')(conv9)

    conv10 = Conv3D(1, (1,1,1))(conv9)

    # model = Model(inputs=[inputs], outputs=[conv10])
    model = Model(inputs=[inputs,gx0,gx1,gx2,gx3], outputs=[conv10])
    
    return model

def cross_entropy_balanced(y_true, y_pred):
    # Note: tf.nn.sigmoid_cross_entropy_with_logits expects y_pred is logits, 
    # Keras expects probabilities.
    # transform y_pred back to logits
    _epsilon = _to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred   = tf.clip_by_value(y_pred, _epsilon, 1 - _epsilon)
    y_pred   = tf.log(y_pred/ (1 - y_pred))

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    beta = count_neg / (count_neg + count_pos)

    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=pos_weight)

    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
    x: An object to be converted (numpy array, list, tensors).
    dtype: The destination type.
    # Returns
    A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x
