# -*- coding: utf-8 -*-
# Generator networks for super-resolution
# LZY 2020
#import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, concatenate, add, Multiply, BatchNormalization, Activation, \
                         MaxPooling3D, UpSampling3D, ELU
                         
def conv3d_bn_relu(inputs, filter_num):
    conv = Conv3D(filter_num, (3,3,3), padding='same', 
                  activation='relu', 
                  kernel_initializer='he_normal')(inputs)
    return conv

def munet_3d_model(num_ch, output_ch, filter_num=64, kinit_type='he_normal', tag='modified_unet3d'):
    
    inputs = Input((None, None, None, num_ch)) 
    loss_weights = Input((None, None, None, 1))
    
    p0 = inputs
    
    conv1 = conv3d_bn_relu(p0, filter_num)
    conv1 = conv3d_bn_relu(conv1, filter_num)
    
    conv2 = conv3d_bn_relu(conv1, filter_num)
    conv2 = conv3d_bn_relu(conv2, filter_num)

    conv3 = conv3d_bn_relu(conv2, filter_num)
    conv3 = conv3d_bn_relu(conv3, filter_num)
   
    conv4 = conv3d_bn_relu(conv3, filter_num)
    conv4 = conv3d_bn_relu(conv4, filter_num)

    conv5 = conv3d_bn_relu(conv4, filter_num)
    conv5 = conv3d_bn_relu(conv5, filter_num)

    merge6 = concatenate([conv4,conv5])
    conv6 = conv3d_bn_relu(merge6, filter_num)
    conv6 = conv3d_bn_relu(conv6, filter_num)
    
    merge7 = concatenate([conv3,conv6])
    conv7 = conv3d_bn_relu(merge7, filter_num)
    conv7 = conv3d_bn_relu(conv7, filter_num)

    merge8 = concatenate([conv2,conv7])
    conv8 = conv3d_bn_relu(merge8, filter_num)
    conv8 = conv3d_bn_relu(conv8, filter_num)

    merge9 = concatenate([conv1,conv8])
    conv9 = conv3d_bn_relu(merge9, filter_num)
    conv9 = conv3d_bn_relu(conv9, filter_num)
    
    residual = Conv3D(output_ch, (3, 3, 3), padding='same',
                  activation=None, 
                  kernel_initializer='he_normal')(conv9)
    
    conv = concatenate([residual, loss_weights],axis=-1)
        
    model = Model(inputs=[inputs, loss_weights], outputs=conv)  
    
    return model