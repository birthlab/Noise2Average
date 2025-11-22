# -*- coding: utf-8 -*-
# Generator networks for super-resolution
# LZY 2020
#import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, concatenate, add, Multiply, BatchNormalization, Activation, \
                         MaxPooling3D, UpSampling3D, ELU
                         
def conv3d_bn_relu(inputs, filter_num, bn_flag=True):
    
    if bn_flag:
        conv = Conv3D(filter_num, (3,3,3), padding='same', kernel_initializer='he_normal')(inputs)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)        
    else:
        conv = Conv3D(filter_num, (3,3,3), padding='same', 
                      activation='relu', 
                      kernel_initializer='he_normal')(inputs)
    return conv

def munet_3d_model(num_ch, output_ch, filter_num=64, kinit_type='he_normal', tag='modified_unet3d'):
    
    inputs = Input((None, None, None, num_ch)) 
    loss_weights = Input((None, None, None, 1))
    
    p0 = inputs
    
    conv1 = conv3d_bn_relu(p0, filter_num, bn_flag=False)
    conv1 = conv3d_bn_relu(conv1, filter_num)
    
    conv2 = conv3d_bn_relu(conv1, filter_num)
    conv2 = conv3d_bn_relu(conv2, filter_num)

    conv3 = conv3d_bn_relu(conv2, filter_num)
    conv3 = conv3d_bn_relu(conv3, filter_num)

    merge4 = concatenate([conv3,conv1])
    conv4 = conv3d_bn_relu(merge4, filter_num)
    conv4 = conv3d_bn_relu(conv4, filter_num)
    
    merge5 = concatenate([conv4,conv2])
    conv5 = conv3d_bn_relu(merge5, filter_num//2)
    conv5 = conv3d_bn_relu(conv5, filter_num//4)
    
    residual = Conv3D(output_ch, (3, 3, 3), padding='same',
                  activation=None, 
                  kernel_initializer='he_normal')(conv5)
    
#    # add residual
#    layer_name = 'add_residual'
#    conv = add([inputs, residual], name=layer_name) # add residual
    
    conv = concatenate([residual, loss_weights],axis=-1)
        
    model = Model(inputs=[inputs, loss_weights], outputs=conv)  
    
    return model