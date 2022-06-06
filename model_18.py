# -*- coding:utf-8 -*-
from model_profiler import model_profiler
import tensorflow as tf

def pool_decoder(input, pool_size=8, filters=64):

    input = tf.keras.layers.MaxPool2D(pool_size, pool_size)(input)
    input = tf.keras.layers.BatchNormalization()(input)
    input = tf.keras.layers.ReLU()(input)
    input = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(input)
    input = tf.keras.layers.BatchNormalization()(input)
    input = tf.keras.layers.ReLU()(input)

    return input

def upsample_decoder(input, upsample=2, filters=64):

    input = tf.keras.layers.UpSampling2D(upsample, interpolation='bilinear')(input)
    input = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(input)
    input = tf.keras.layers.BatchNormalization()(input)
    input = tf.keras.layers.ReLU()(input)

    return input

def conv_decoder(input, filters):

    input = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(input)
    input = tf.keras.layers.BatchNormalization()(input)
    input = tf.keras.layers.ReLU()(input)

    return input

def modified_Unet_patch(input_shape=(576, 576, 3)):

    Catchannel = 32
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
    h_5 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", use_bias=False)(h_5)
    h_5 = tf.keras.layers.BatchNormalization()(h_5)
    h_5 = tf.keras.layers.ReLU()(h_5)

    h_5_1 = h_5[:, 0:18, 0:18, :]
    h_5_2 = h_5[:, 0:18, 18:, :]
    h_5_3 = h_5[:, 18:, 0:18, :]
    h_5_4 = h_5[:, 18:, 18:, :]

    h_4_1 = h_4[:, 0:36, 0:36, :]
    h_4_2 = h_4[:, 0:36, 36:, :]
    h_4_3 = h_4[:, 36:, 0:36, :]
    h_4_4 = h_4[:, 36:, 36:, :]

    h_3_1 = h_3[:, 0:72, 0:72, :]
    h_3_2 = h_3[:, 0:72, 72:, :]
    h_3_3 = h_3[:, 72:, 0:72, :]
    h_3_4 = h_3[:, 72:, 72:, :]

    h_2_1 = h_2[:, 0:144, 0:144, :]
    h_2_2 = h_2[:, 0:144, 144:, :]
    h_2_3 = h_2[:, 144:, 0:144, :]
    h_2_4 = h_2[:, 144:, 144:, :]

    h_1_1 = h_1[:, 0:288, 0:288, :]
    h_1_2 = h_1[:, 0:288, 288:, :]
    h_1_3 = h_1[:, 288:, 0:288, :]
    h_1_4 = h_1[:, 288:, 288:, :]

    ###################################################################################################
    h_1_1_PT_hd4 = pool_decoder(h_1_1, 8, Catchannel)
    h_1_2_PT_hd4 = pool_decoder(h_1_2, 8, Catchannel)
    h_1_3_PT_hd4 = pool_decoder(h_1_3, 8, Catchannel)
    h_1_4_PT_hd4 = pool_decoder(h_1_4, 8, Catchannel)

    h_2_1_PT_hd4 = pool_decoder(h_2_1, 4, Catchannel)
    h_2_2_PT_hd4 = pool_decoder(h_2_2, 4, Catchannel)
    h_2_3_PT_hd4 = pool_decoder(h_2_3, 4, Catchannel)
    h_2_4_PT_hd4 = pool_decoder(h_2_4, 4, Catchannel)

    h_3_1_PT_hd4 = pool_decoder(h_3_1, 2, Catchannel)
    h_3_2_PT_hd4 = pool_decoder(h_3_2, 2, Catchannel)
    h_3_3_PT_hd4 = pool_decoder(h_3_3, 2, Catchannel)
    h_3_4_PT_hd4 = pool_decoder(h_3_4, 2, Catchannel)

    h_4_1_Cat_hd4 = conv_decoder(h_4_1, Catchannel)
    h_4_2_Cat_hd4 = conv_decoder(h_4_2, Catchannel)
    h_4_3_Cat_hd4 = conv_decoder(h_4_3, Catchannel)
    h_4_4_Cat_hd4 = conv_decoder(h_4_4, Catchannel)

    h_5_1_PT_hd4 = upsample_decoder(h_5_1, 2, Catchannel)
    h_5_2_PT_hd4 = upsample_decoder(h_5_2, 2, Catchannel)
    h_5_3_PT_hd4 = upsample_decoder(h_5_3, 2, Catchannel)
    h_5_4_PT_hd4 = upsample_decoder(h_5_4, 2, Catchannel)

    hd4_1 = tf.concat([h_1_1_PT_hd4, h_2_1_PT_hd4, h_3_1_PT_hd4, h_4_1_Cat_hd4, h_5_1_PT_hd4], -1)
    hd4_1 = conv_decoder(hd4_1, UpChannel)
    hd4_2 = tf.concat([h_1_2_PT_hd4, h_2_2_PT_hd4, h_3_2_PT_hd4, h_4_2_Cat_hd4, h_5_2_PT_hd4], -1)
    hd4_2 = conv_decoder(hd4_2, UpChannel)
    hd4_3 = tf.concat([h_1_3_PT_hd4, h_2_3_PT_hd4, h_3_3_PT_hd4, h_4_3_Cat_hd4, h_5_3_PT_hd4], -1)
    hd4_3 = conv_decoder(hd4_3, UpChannel)
    hd4_4 = tf.concat([h_1_4_PT_hd4, h_2_4_PT_hd4, h_3_4_PT_hd4, h_4_4_Cat_hd4, h_5_4_PT_hd4], -1)
    hd4_4 = conv_decoder(hd4_4, UpChannel)
    ###################################################################################################

    ###################################################################################################
    h_1_1_PT_hd3 = upsample_decoder(h_1_1_PT_hd4, 2, Catchannel)
    h_1_2_PT_hd3 = upsample_decoder(h_1_2_PT_hd4, 2, Catchannel)
    h_1_3_PT_hd3 = upsample_decoder(h_1_3_PT_hd4, 2, Catchannel)
    h_1_4_PT_hd3 = upsample_decoder(h_1_4_PT_hd4, 2, Catchannel)

    h_2_1_PT_hd3 = upsample_decoder(h_2_1_PT_hd4, 2, Catchannel)
    h_2_2_PT_hd3 = upsample_decoder(h_2_2_PT_hd4, 2, Catchannel)
    h_2_3_PT_hd3 = upsample_decoder(h_2_3_PT_hd4, 2, Catchannel)
    h_2_4_PT_hd3 = upsample_decoder(h_2_4_PT_hd4, 2, Catchannel)

    h_3_1_PT_hd3 = upsample_decoder(h_3_1_PT_hd4, 2, Catchannel)
    h_3_2_PT_hd3 = upsample_decoder(h_3_2_PT_hd4, 2, Catchannel)
    h_3_3_PT_hd3 = upsample_decoder(h_3_3_PT_hd4, 2, Catchannel)
    h_3_4_PT_hd3 = upsample_decoder(h_3_4_PT_hd4, 2, Catchannel)

    hd4_1_UT_hd3 = upsample_decoder(hd4_1, 2, Catchannel)
    hd4_2_UT_hd3 = upsample_decoder(hd4_2, 2, Catchannel)
    hd4_3_UT_hd3 = upsample_decoder(hd4_3, 2, Catchannel)
    hd4_4_UT_hd3 = upsample_decoder(hd4_4, 2, Catchannel)

    hd5_1_UT_hd3 = upsample_decoder(h_5_1_PT_hd4, 2, Catchannel)
    hd5_2_UT_hd3 = upsample_decoder(h_5_2_PT_hd4, 2, Catchannel)
    hd5_3_UT_hd3 = upsample_decoder(h_5_3_PT_hd4, 2, Catchannel)
    hd5_4_UT_hd3 = upsample_decoder(h_5_4_PT_hd4, 2, Catchannel)

    hd3_1 = tf.concat([h_1_1_PT_hd3, h_2_1_PT_hd3, h_3_1_PT_hd3, hd4_1_UT_hd3, hd5_1_UT_hd3], -1)
    hd3_1 = conv_decoder(hd3_1, UpChannel)
    hd3_2 = tf.concat([h_1_2_PT_hd3, h_2_2_PT_hd3, h_3_2_PT_hd3, hd4_2_UT_hd3, hd5_2_UT_hd3], -1)
    hd3_2 = conv_decoder(hd3_2, UpChannel)
    hd3_3 = tf.concat([h_1_3_PT_hd3, h_2_3_PT_hd3, h_3_3_PT_hd3, hd4_3_UT_hd3, hd5_3_UT_hd3], -1)
    hd3_3 = conv_decoder(hd3_3, UpChannel)
    hd3_4 = tf.concat([h_1_4_PT_hd3, h_2_4_PT_hd3, h_3_4_PT_hd3, hd4_4_UT_hd3, hd5_4_UT_hd3], -1)
    hd3_4 = conv_decoder(hd3_4, UpChannel)
    ###################################################################################################

    ###################################################################################################
    h_1_1_PT_hd2 = upsample_decoder(h_1_1_PT_hd3, 2, Catchannel)
    h_1_2_PT_hd2 = upsample_decoder(h_1_2_PT_hd3, 2, Catchannel)
    h_1_3_PT_hd2 = upsample_decoder(h_1_3_PT_hd3, 2, Catchannel)
    h_1_4_PT_hd2 = upsample_decoder(h_1_4_PT_hd3, 2, Catchannel)

    h_2_1_PT_hd2 = upsample_decoder(h_2_1_PT_hd3, 2, Catchannel)
    h_2_2_PT_hd2 = upsample_decoder(h_2_2_PT_hd3, 2, Catchannel)
    h_2_3_PT_hd2 = upsample_decoder(h_2_3_PT_hd3, 2, Catchannel)
    h_2_4_PT_hd2 = upsample_decoder(h_2_4_PT_hd3, 2, Catchannel)

    h_3_1_PT_hd2 = upsample_decoder(hd3_1, 2, Catchannel)
    h_3_2_PT_hd2 = upsample_decoder(hd3_2, 2, Catchannel)
    h_3_3_PT_hd2 = upsample_decoder(hd3_3, 2, Catchannel)
    h_3_4_PT_hd2 = upsample_decoder(hd3_4, 2, Catchannel)

    h_4_1_PT_hd2 = upsample_decoder(hd4_1, 4, Catchannel)
    h_4_2_PT_hd2 = upsample_decoder(hd4_2, 4, Catchannel)
    h_4_3_PT_hd2 = upsample_decoder(hd4_3, 4, Catchannel)
    h_4_4_PT_hd2 = upsample_decoder(hd4_4, 4, Catchannel)

    h_5_1_PT_hd2 = upsample_decoder(hd5_1_UT_hd3, 2, Catchannel)
    h_5_2_PT_hd2 = upsample_decoder(hd5_2_UT_hd3, 2, Catchannel)
    h_5_3_PT_hd2 = upsample_decoder(hd5_3_UT_hd3, 2, Catchannel)
    h_5_4_PT_hd2 = upsample_decoder(hd5_4_UT_hd3, 2, Catchannel)

    hd2_1 = tf.concat([h_1_1_PT_hd2, h_2_1_PT_hd2, h_3_1_PT_hd2, h_4_1_PT_hd2, h_5_1_PT_hd2], -1)
    hd2_1 = conv_decoder(hd2_1, UpChannel)
    hd2_2 = tf.concat([h_1_2_PT_hd2, h_2_2_PT_hd2, h_3_2_PT_hd2, h_4_2_PT_hd2, h_5_2_PT_hd2], -1)
    hd2_2 = conv_decoder(hd2_2, UpChannel)
    hd2_3 = tf.concat([h_1_3_PT_hd2, h_2_3_PT_hd2, h_3_3_PT_hd2, h_4_3_PT_hd2, h_5_3_PT_hd2], -1)
    hd2_3 = conv_decoder(hd2_3, UpChannel)
    hd2_4 = tf.concat([h_1_4_PT_hd2, h_2_4_PT_hd2, h_3_4_PT_hd2, h_4_4_PT_hd2, h_5_4_PT_hd2], -1)
    hd2_4 = conv_decoder(hd2_4, UpChannel)
    ###################################################################################################

    ###################################################################################################
    h_1_1_PT_hd1 = upsample_decoder(h_1_1_PT_hd2, 2, Catchannel)
    h_1_2_PT_hd1 = upsample_decoder(h_1_2_PT_hd2, 2, Catchannel)
    h_1_3_PT_hd1 = upsample_decoder(h_1_3_PT_hd2, 2, Catchannel)
    h_1_4_PT_hd1 = upsample_decoder(h_1_4_PT_hd2, 2, Catchannel)

    h_2_1_PT_hd1 = upsample_decoder(hd2_1, 2, Catchannel)
    h_2_2_PT_hd1 = upsample_decoder(hd2_2, 2, Catchannel)
    h_2_3_PT_hd1 = upsample_decoder(hd2_3, 2, Catchannel)
    h_2_4_PT_hd1 = upsample_decoder(hd2_4, 2, Catchannel)

    h_3_1_PT_hd1 = upsample_decoder(hd3_1, 4, Catchannel)
    h_3_2_PT_hd1 = upsample_decoder(hd3_2, 4, Catchannel)
    h_3_3_PT_hd1 = upsample_decoder(hd3_3, 4, Catchannel)
    h_3_4_PT_hd1 = upsample_decoder(hd3_4, 4, Catchannel)

    h_4_1_PT_hd1 = upsample_decoder(hd4_1, 8, Catchannel)
    h_4_2_PT_hd1 = upsample_decoder(hd4_2, 8, Catchannel)
    h_4_3_PT_hd1 = upsample_decoder(hd4_3, 8, Catchannel)
    h_4_4_PT_hd1 = upsample_decoder(hd4_4, 8, Catchannel)

    h_5_1_PT_hd1 = upsample_decoder(h_5_1_PT_hd2, 2, Catchannel)
    h_5_2_PT_hd1 = upsample_decoder(h_5_2_PT_hd2, 2, Catchannel)
    h_5_3_PT_hd1 = upsample_decoder(h_5_3_PT_hd2, 2, Catchannel)
    h_5_4_PT_hd1 = upsample_decoder(h_5_4_PT_hd2, 2, Catchannel)

    hd1_1 = tf.concat([h_1_1_PT_hd1, h_2_1_PT_hd1, h_3_1_PT_hd1, h_4_1_PT_hd1, h_5_1_PT_hd1], -1)
    hd1_1 = conv_decoder(hd1_1, UpChannel)
    hd1_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(hd1_1)
    hd1_2 = tf.concat([h_1_2_PT_hd1, h_2_2_PT_hd1, h_3_2_PT_hd1, h_4_2_PT_hd1, h_5_2_PT_hd1], -1)
    hd1_2 = conv_decoder(hd1_2, UpChannel)
    hd1_2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(hd1_2)
    hd1_3 = tf.concat([h_1_3_PT_hd1, h_2_3_PT_hd1, h_3_3_PT_hd1, h_4_3_PT_hd1, h_5_3_PT_hd1], -1)
    hd1_3 = conv_decoder(hd1_3, UpChannel)
    hd1_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(hd1_3)
    hd1_4 = tf.concat([h_1_4_PT_hd1, h_2_4_PT_hd1, h_3_4_PT_hd1, h_4_4_PT_hd1, h_5_4_PT_hd1], -1)
    hd1_4 = conv_decoder(hd1_4, UpChannel)
    hd1_4 = tf.keras.layers.Conv2D(filters=1, kernel_size=1)(hd1_4)
    ###################################################################################################

    return tf.keras.Model(inputs=inputs, outputs=[hd1_1, hd1_2, hd1_3, hd1_4])

mo = modified_Unet_patch()
prob = model_profiler(mo, 3)
mo.summary()
print(prob)
