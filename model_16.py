# -*- coding:utf-8 -*-
from model_profiler import model_profiler

import tensorflow as tf

def dotProduct(input, cla):

    B, H, W, C = input.shape

    input = tf.reshape(input, [-1, C, H * W])
    final = tf.einsum("ijk,ij->ijk", input, cla)
    final = tf.reshape(final, [-1, H, W, C])

    return final

def Unet3_plus_modified(input_shape=(512, 512, 3), nclasses=1):

    Catchannel = 64
    CatBlock = 5
    UpChannel = Catchannel * CatBlock

    h = inputs = tf.keras.Input(input_shape)

    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_1)
    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", use_bias=False)(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_2)
    h_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)
    h_3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", use_bias=False)(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_3)
    h_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)
    h_4 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_4)
    h_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_5)
    h_5 = tf.keras.layers.BatchNormalization()(h_5)
    h_5 = tf.keras.layers.ReLU()(h_5)
    h_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_5)
    h_5 = tf.keras.layers.BatchNormalization()(h_5)
    h_5 = tf.keras.layers.ReLU()(h_5)

    cls_branch = tf.keras.layers.SpatialDropout2D(0.5)(h_5)
    cls_branch = tf.keras.layers.Conv2D(filters=2, kernel_size=1)(cls_branch)
    cls_branch = tf.keras.layers.GlobalMaxPool2D()(cls_branch)
    cls_branch = tf.nn.sigmoid(cls_branch)
    cls_branch_max = tf.keras.backend.max(cls_branch, -1)

    #####################################################################################
    h1_PT_hd4 = tf.keras.layers.MaxPool2D(8, 8)(h_1)
    h1_PT_hd4 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h1_PT_hd4)
    h_4_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_4)
    h_4_ = tf.where(tf.nn.sigmoid(h_4_) >= 0.5, h_4_, 0.)
    h1_PT_hd4 = tf.where(tf.nn.sigmoid(h1_PT_hd4) >= 0.5, h1_PT_hd4, 0.)
    h1_PT_hd4 = tf.keras.layers.BatchNormalization()(h1_PT_hd4 + h_4_)
    h1_PT_hd4 = tf.keras.layers.ReLU()(h1_PT_hd4)

    h2_PT_hd4 = tf.keras.layers.MaxPool2D(4, 4)(h_2)
    h2_PT_hd4 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h2_PT_hd4)
    h_4_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_4)
    h_4_ = tf.where(tf.nn.sigmoid(h_4_) >= 0.5, h_4_, 0.)
    h2_PT_hd4 = tf.where(tf.nn.sigmoid(h2_PT_hd4) >= 0.5, h2_PT_hd4, 0.)
    h2_PT_hd4 = tf.keras.layers.BatchNormalization()(h2_PT_hd4 + h_4_)
    h2_PT_hd4 = tf.keras.layers.ReLU()(h2_PT_hd4)

    h3_PT_hd4 = tf.keras.layers.MaxPool2D(2, 2)(h_3)
    h3_PT_hd4 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h3_PT_hd4)
    h_4_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_4)
    h_4_ = tf.where(tf.nn.sigmoid(h_4_) >= 0.5, h_4_, 0.)
    h3_PT_hd4 = tf.where(tf.nn.sigmoid(h3_PT_hd4) >= 0.5, h3_PT_hd4, 0.)
    h3_PT_hd4 = tf.keras.layers.BatchNormalization()(h3_PT_hd4 + h_4_)
    h3_PT_hd4 = tf.keras.layers.ReLU()(h3_PT_hd4)

    h4_Cat_hd4 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_4)
    h4_Cat_hd4 = tf.keras.layers.BatchNormalization()(h4_Cat_hd4)
    h4_Cat_hd4 = tf.keras.layers.ReLU()(h4_Cat_hd4)

    hd5_UT_hd4 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(h_5)
    hd5_UT_hd4 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd5_UT_hd4)
    h_4_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_4)
    h_4_ = tf.where(tf.nn.sigmoid(h_4_) >= 0.5, h_4_, 0.)
    hd5_UT_hd4 = tf.where(tf.nn.sigmoid(hd5_UT_hd4) >= 0.5, hd5_UT_hd4, 0.)
    hd5_UT_hd4 = tf.keras.layers.BatchNormalization()(hd5_UT_hd4 + h_4_)
    hd5_UT_hd4 = tf.keras.layers.ReLU()(hd5_UT_hd4)

    hd4 = tf.concat([h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4], -1)
    hd4 = tf.keras.layers.Conv2D(filters=UpChannel, kernel_size=3, padding="same", use_bias=False)(hd4)
    hd4 = tf.keras.layers.BatchNormalization()(hd4)
    hd4 = tf.keras.layers.ReLU()(hd4)
    #####################################################################################

    #####################################################################################
    h1_PT_hd3 = tf.keras.layers.MaxPool2D(4, 4)(h_1)
    h1_PT_hd3 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h1_PT_hd3)
    h_3_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_3)
    h_3_ = tf.where(tf.nn.sigmoid(h_3_) >= 0.5, h_3_, 0.)
    h1_PT_hd3 = tf.where(tf.nn.sigmoid(h1_PT_hd3) >= 0.5, h1_PT_hd3, 0.)
    h1_PT_hd3 = tf.keras.layers.BatchNormalization()(h1_PT_hd3 + h_3_)
    h1_PT_hd3 = tf.keras.layers.ReLU()(h1_PT_hd3)

    h2_PT_hd3 = tf.keras.layers.MaxPool2D(2, 2)(h_2)
    h2_PT_hd3 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h2_PT_hd3)
    h_3_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_3)
    h_3_ = tf.where(tf.nn.sigmoid(h_3_) >= 0.5, h_3_, 0.)
    h2_PT_hd3 = tf.where(tf.nn.sigmoid(h2_PT_hd3) >= 0.5, h2_PT_hd3, 0.)
    h2_PT_hd3 = tf.keras.layers.BatchNormalization()(h2_PT_hd3 + h_3_)
    h2_PT_hd3 = tf.keras.layers.ReLU()(h2_PT_hd3)

    h3_Cat_hd3 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_3)
    h3_Cat_hd3 = tf.keras.layers.BatchNormalization()(h3_Cat_hd3)
    h3_Cat_hd3 = tf.keras.layers.ReLU()(h3_Cat_hd3)

    hd4_UT_hd3 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(hd4)
    hd4_UT_hd3 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd4_UT_hd3)
    h_3_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_3)
    h_3_ = tf.where(tf.nn.sigmoid(h_3_) >= 0.5, h_3_, 0.)
    hd4_UT_hd3 = tf.where(tf.nn.sigmoid(hd4_UT_hd3) >= 0.5, hd4_UT_hd3, 0.)
    hd4_UT_hd3 = tf.keras.layers.BatchNormalization()(hd4_UT_hd3 + h_3_)
    hd4_UT_hd3 = tf.keras.layers.ReLU()(hd4_UT_hd3)

    hd5_UT_hd3 = tf.keras.layers.UpSampling2D(4, interpolation='bilinear')(h_5)
    hd5_UT_hd3 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd5_UT_hd3)
    h_3_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_3)
    h_3_ = tf.where(tf.nn.sigmoid(h_3_) >= 0.5, h_3_, 0.)
    hd5_UT_hd3 = tf.where(tf.nn.sigmoid(hd5_UT_hd3) >= 0.5, hd5_UT_hd3, 0.)
    hd5_UT_hd3 = tf.keras.layers.BatchNormalization()(hd5_UT_hd3 + h_3_)
    hd5_UT_hd3 = tf.keras.layers.ReLU()(hd5_UT_hd3)

    hd3 = tf.concat([h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3], -1)
    hd3 = tf.keras.layers.Conv2D(filters=UpChannel, kernel_size=3, padding="same", use_bias=False)(hd3)
    hd3 = tf.keras.layers.BatchNormalization()(hd3)
    hd3 = tf.keras.layers.ReLU()(hd3)
    #####################################################################################

    #####################################################################################
    h1_PT_hd2 = tf.keras.layers.MaxPool2D(2, 2)(h_1)
    h1_PT_hd2 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same")(h1_PT_hd2)
    h_2_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_2)
    h_2_ = tf.where(tf.nn.sigmoid(h_2_) >= 0.5, h_2_, 0.)
    h1_PT_hd2 = tf.where(tf.nn.sigmoid(h1_PT_hd2) >= 0.5, h1_PT_hd2, 0.)
    h1_PT_hd2 = tf.keras.layers.BatchNormalization()(h1_PT_hd2 + h_2_)
    h1_PT_hd2 = tf.keras.layers.ReLU()(h1_PT_hd2)

    h2_Cat_hd2 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same")(h_2)
    h2_Cat_hd2 = tf.keras.layers.BatchNormalization()(h2_Cat_hd2)
    h2_Cat_hd2 = tf.keras.layers.ReLU()(h2_Cat_hd2)

    hd3_UT_hd2 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(hd3)
    hd3_UT_hd2 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd3_UT_hd2)
    h_2_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_2)
    h_2_ = tf.where(tf.nn.sigmoid(h_2_) >= 0.5, h_2_, 0.)
    hd3_UT_hd2 = tf.where(tf.nn.sigmoid(hd3_UT_hd2) >= 0.5, hd3_UT_hd2, 0.)
    hd3_UT_hd2 = tf.keras.layers.BatchNormalization()(hd3_UT_hd2 + h_2_)
    hd3_UT_hd2 = tf.keras.layers.ReLU()(hd3_UT_hd2)

    hd4_UT_hd2 = tf.keras.layers.UpSampling2D(4, interpolation='bilinear')(hd4)
    hd4_UT_hd2 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd4_UT_hd2)
    h_2_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_2)
    h_2_ = tf.where(tf.nn.sigmoid(h_2_) >= 0.5, h_2_, 0.)
    hd4_UT_hd2 = tf.where(tf.nn.sigmoid(hd4_UT_hd2) >= 0.5, hd4_UT_hd2, 0.)
    hd4_UT_hd2 = tf.keras.layers.BatchNormalization()(hd4_UT_hd2 + h_2_)
    hd4_UT_hd2 = tf.keras.layers.ReLU()(hd4_UT_hd2)

    hd5_UT_hd2 = tf.keras.layers.UpSampling2D(8, interpolation='bilinear')(h_5)
    hd5_UT_hd2 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd5_UT_hd2)
    h_2_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_2)
    h_2_ = tf.where(tf.nn.sigmoid(h_2_) >= 0.5, h_2_, 0.)
    hd5_UT_hd2 = tf.where(tf.nn.sigmoid(hd5_UT_hd2) >= 0.5, hd5_UT_hd2, 0.)
    hd5_UT_hd2 = tf.keras.layers.BatchNormalization()(hd5_UT_hd2 + h_2_)
    hd5_UT_hd2 = tf.keras.layers.ReLU()(hd5_UT_hd2)

    hd2 = tf.concat([h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2], -1)
    hd2 = tf.keras.layers.Conv2D(filters=UpChannel, kernel_size=3, padding="same", use_bias=False)(hd2)
    hd2 = tf.keras.layers.BatchNormalization()(hd2)
    hd2 = tf.keras.layers.ReLU()(hd2)
    #####################################################################################

    #####################################################################################
    h1_Cat_hd1 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_1)
    h1_Cat_hd1 = tf.keras.layers.BatchNormalization()(h1_Cat_hd1)
    h1_Cat_hd1 = tf.keras.layers.ReLU()(h1_Cat_hd1)

    hd2_UT_hd1 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(hd2)
    hd2_UT_hd1 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd2_UT_hd1)
    h_1_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_ = tf.where(tf.nn.sigmoid(h_1_) >= 0.5, h_1_, 0.)
    hd2_UT_hd1 = tf.where(tf.nn.sigmoid(hd2_UT_hd1) >= 0.5, hd2_UT_hd1, 0.)
    hd2_UT_hd1 = tf.keras.layers.BatchNormalization()(hd2_UT_hd1 + h_1_)
    hd2_UT_hd1 = tf.keras.layers.ReLU()(hd2_UT_hd1)

    hd3_UT_hd1 = tf.keras.layers.UpSampling2D(4, interpolation='bilinear')(hd3)
    hd3_UT_hd1 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd3_UT_hd1)
    h_1_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_ = tf.where(tf.nn.sigmoid(h_1_) >= 0.5, h_1_, 0.)
    hd3_UT_hd1 = tf.where(tf.nn.sigmoid(hd3_UT_hd1) >= 0.5, hd3_UT_hd1, 0.)
    hd3_UT_hd1 = tf.keras.layers.BatchNormalization()(hd3_UT_hd1 + h_1_)
    hd3_UT_hd1 = tf.keras.layers.ReLU()(hd3_UT_hd1)

    hd4_UT_hd1 = tf.keras.layers.UpSampling2D(8, interpolation='bilinear')(hd4)
    hd4_UT_hd1 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd4_UT_hd1)
    h_1_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_ = tf.where(tf.nn.sigmoid(h_1_) >= 0.5, h_1_, 0.)
    hd4_UT_hd1 = tf.where(tf.nn.sigmoid(hd4_UT_hd1) >= 0.5, hd4_UT_hd1, 0.)
    hd4_UT_hd1 = tf.keras.layers.BatchNormalization()(hd4_UT_hd1 + h_1_)
    hd4_UT_hd1 = tf.keras.layers.ReLU()(hd4_UT_hd1)

    hd5_UT_hd1 = tf.keras.layers.UpSampling2D(16, interpolation='bilinear')(h_5)
    hd5_UT_hd1 = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(hd5_UT_hd1)
    h_1_ = tf.keras.layers.Conv2D(filters=Catchannel, kernel_size=3, padding="same", use_bias=False)(h_1)
    h_1_ = tf.where(tf.nn.sigmoid(h_1_) >= 0.5, h_1_, 0.)
    hd5_UT_hd1 = tf.where(tf.nn.sigmoid(hd5_UT_hd1) >= 0.5, hd5_UT_hd1, 0.)
    hd5_UT_hd1 = tf.keras.layers.BatchNormalization()(hd5_UT_hd1 + h_1_)
    hd5_UT_hd1 = tf.keras.layers.ReLU()(hd5_UT_hd1)

    hd1 = tf.concat([h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1], -1)
    hd1 = tf.keras.layers.Conv2D(filters=UpChannel, kernel_size=3, padding="same", use_bias=False)(hd1)
    hd1 = tf.keras.layers.BatchNormalization()(hd1)
    hd1 = tf.keras.layers.ReLU()(hd1)

    d5 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=3, padding="same")(h_5)
    d5 = tf.keras.layers.UpSampling2D(16, interpolation='bilinear')(d5)

    d4 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=3, padding="same")(hd4)
    d4 = tf.keras.layers.UpSampling2D(8, interpolation='bilinear')(d4)

    d3 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=3, padding="same")(hd3)
    d3 = tf.keras.layers.UpSampling2D(4, interpolation='bilinear')(d3)

    d2 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=3, padding="same")(hd2)
    d2 = tf.keras.layers.UpSampling2D(2, interpolation='bilinear')(d2)

    d1 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=3, padding="same")(hd1)

    # d1 = dotProduct(d1, cls_branch_max) # Pixel attention   # this part is problem!!!!!!!!!!
    # d2 = dotProduct(d2, cls_branch_max) # Pixel attention
    # d3 = dotProduct(d3, cls_branch_max) # Pixel attention
    # d4 = dotProduct(d4, cls_branch_max) # Pixel attention
    # d5 = dotProduct(d5, cls_branch_max) # Pixel attention

    d1 = tf.keras.layers.multiply([d1, cls_branch_max])
    d2 = tf.keras.layers.multiply([d2, cls_branch_max])
    d3 = tf.keras.layers.multiply([d3, cls_branch_max])
    d4 = tf.keras.layers.multiply([d4, cls_branch_max])
    d5 = tf.keras.layers.multiply([d5, cls_branch_max])

    return tf.keras.Model(inputs=inputs, outputs=[d1,
                                                  d2,
                                                  d3,
                                                  d4,
                                                  d5,
                                                  cls_branch_max])
