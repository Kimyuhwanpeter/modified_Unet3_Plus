# -*- coding:utf-8 -*-
from model_profiler import model_profiler
import tensorflow as tf
rr = tf.keras.applications.ResNet152
def res_connection(input, filters, conv_shortcut=True):

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(
        filters,
        1,
        strides=1,
        use_bias=False)(input)
        shortcut = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(shortcut)
    else:
        shortcut = input

    h = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=1, use_bias=False)(input)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)

    h = tf.keras.layers.Add()([shortcut, h])
    h = tf.keras.layers.ReLU()(h)

    return h

def decoder_res_connection(input, filters, conv_shortcut=True):

    if conv_shortcut:
        shortcut = tf.keras.layers.Conv2D(
        filters // 4,
        1,
        strides=1,
        use_bias=False)(input)
        shortcut = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(shortcut)
    else:
        shortcut = input

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, use_bias=False)(input)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, padding="same", use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=filters // 4, kernel_size=1, use_bias=False)(h)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)

    h = tf.keras.layers.Add()([shortcut, h])
    h = tf.keras.layers.ReLU()(h)

    return h

def conv_bn_sigmoid(input, filters):
    h = tf.keras.layers.Conv2D(filters=filters, kernel_size=1, use_bias=False)(input)
    h = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h)
    h = tf.keras.layers.Activation('sigmoid')(h)
    return h

def patch_flow_Unet(input_shape=(320, 320, 3), nclasses=1):

    dim = 64
    h = inputs = tf.keras.Input(input_shape)

    h_1 = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, padding="same", use_bias=False, name="block1_conv1")(h)
    h_1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1 = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, padding="same", use_bias=False, name="block1_conv2")(h_1)
    h_1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)
    h_1_cat_block = h_1

    h_1_1 = h_1[:, 0:input_shape[0] // 2, 0:input_shape[1] // 2, :]
    h_1_2 = h_1[:, 0:input_shape[0] // 2, input_shape[1] // 2:, :]
    h_1_3 = h_1[:, input_shape[0] // 2:, 0:input_shape[1] // 2, :]
    h_1_4 = h_1[:, input_shape[0] // 2:, input_shape[1] // 2:, :]

    h_1_1 = res_connection(h_1_1, dim, True)
    h_1_2 = res_connection(h_1_2, dim, True)
    h_1_3 = res_connection(h_1_3, dim, True)
    h_1_4 = res_connection(h_1_4, dim, True)
    for _ in range(2):
        h_1_1 = res_connection(h_1_1, dim, False)
        h_1_2 = res_connection(h_1_2, dim, False)
        h_1_3 = res_connection(h_1_3, dim, False)
        h_1_4 = res_connection(h_1_4, dim, False)
    h_1_1_cat_block = h_1_1
    h_1_2_cat_block = h_1_2
    h_1_3_cat_block = h_1_3
    h_1_4_cat_block = h_1_4

    h_1_1 = conv_bn_sigmoid(h_1_1, dim)
    h_1_2 = conv_bn_sigmoid(h_1_2, dim)
    h_1_3 = conv_bn_sigmoid(h_1_3, dim)
    h_1_4 = conv_bn_sigmoid(h_1_4, dim)

    h_1_1 = h_1[:, 0:input_shape[0] // 2, 0:input_shape[1] // 2, :] * h_1_1
    h_1_2 = h_1[:, 0:input_shape[0] // 2, input_shape[1] // 2:, :] * h_1_2
    h_1_3 = h_1[:, input_shape[0] // 2:, 0:input_shape[1] // 2, :] * h_1_3
    h_1_4 = h_1[:, input_shape[0] // 2:, input_shape[1] // 2:, :] * h_1_4
    
    h_1_12 = tf.concat([h_1_1, h_1_2], 2)
    h_1_34 = tf.concat([h_1_3, h_1_4], 2)
    h_1 = tf.concat([h_1_12, h_1_34], 1)

    h_2 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_1)
    h_2 = tf.keras.layers.Conv2D(filters=dim*2, kernel_size=3, padding="same", use_bias=False, name="block2_conv1")(h_2)
    h_2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2 = tf.keras.layers.Conv2D(filters=dim*2, kernel_size=3, padding="same", use_bias=False, name="block2_conv2")(h_2)
    h_2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)
    h_2_cat_block = h_2

    h_2_1 = h_2[:, 0:input_shape[0] // 4, 0:input_shape[1] // 4, :]
    h_2_2 = h_2[:, 0:input_shape[0] // 4, input_shape[1] // 4:, :]
    h_2_3 = h_2[:, input_shape[0] // 4:, 0:input_shape[1] // 4, :]
    h_2_4 = h_2[:, input_shape[0] // 4:, input_shape[1] // 4:, :]

    h_2_1 = res_connection(h_2_1, dim*2, True)
    h_2_2 = res_connection(h_2_2, dim*2, True)
    h_2_3 = res_connection(h_2_3, dim*2, True)
    h_2_4 = res_connection(h_2_4, dim*2, True)
    for _ in range(2):
        h_2_1 = res_connection(h_2_1, dim*2, False)
        h_2_2 = res_connection(h_2_2, dim*2, False)
        h_2_3 = res_connection(h_2_3, dim*2, False)
        h_2_4 = res_connection(h_2_4, dim*2, False)
    h_2_1_cat_block = h_2_1
    h_2_2_cat_block = h_2_2
    h_2_3_cat_block = h_2_3
    h_2_4_cat_block = h_2_4

    h_2_1 = conv_bn_sigmoid(h_2_1, dim*2)
    h_2_2 = conv_bn_sigmoid(h_2_2, dim*2)
    h_2_3 = conv_bn_sigmoid(h_2_3, dim*2)
    h_2_4 = conv_bn_sigmoid(h_2_4, dim*2)

    h_2_1 = h_2[:, 0:input_shape[0] // 4, 0:input_shape[1] // 4, :] * h_2_1
    h_2_2 = h_2[:, 0:input_shape[0] // 4, input_shape[1] // 4:, :] * h_2_2
    h_2_3 = h_2[:, input_shape[0] // 4:, 0:input_shape[1] // 4, :] * h_2_3
    h_2_4 = h_2[:, input_shape[0] // 4:, input_shape[1] // 4:, :] * h_2_4
    
    h_2_12 = tf.concat([h_2_1, h_2_2], 2)
    h_2_34 = tf.concat([h_2_3, h_2_4], 2)
    h_2 = tf.concat([h_2_12, h_2_34], 1)

    h_3 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_2)
    h_3 = tf.keras.layers.Conv2D(filters=dim*4, kernel_size=3, padding="same", use_bias=False, name="block3_conv1")(h_3)
    h_3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)
    h_3 = tf.keras.layers.Conv2D(filters=dim*4, kernel_size=3, padding="same", use_bias=False, name="block3_conv2")(h_3)
    h_3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)
    h_3 = tf.keras.layers.Conv2D(filters=dim*4, kernel_size=3, padding="same", use_bias=False, name="block3_conv3")(h_3)
    h_3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)
    h_3_cat_block = h_3

    h_3_1 = h_3[:, 0:input_shape[0] // 8, 0:input_shape[1] // 8 :]
    h_3_2 = h_3[:, 0:input_shape[0] // 8, input_shape[1] // 8:, :]
    h_3_3 = h_3[:, input_shape[0] // 8:, 0:input_shape[1] // 8, :]
    h_3_4 = h_3[:, input_shape[0] // 8:, input_shape[1] // 8:, :]

    h_3_1 = res_connection(h_3_1, dim*4, True)
    h_3_2 = res_connection(h_3_2, dim*4, True)
    h_3_3 = res_connection(h_3_3, dim*4, True)
    h_3_4 = res_connection(h_3_4, dim*4, True)
    for _ in range(3):
        h_3_1 = res_connection(h_3_1, dim*4, False)
        h_3_2 = res_connection(h_3_2, dim*4, False)
        h_3_3 = res_connection(h_3_3, dim*4, False)
        h_3_4 = res_connection(h_3_4, dim*4, False)
    h_3_1_cat_block = h_3_1
    h_3_2_cat_block = h_3_2
    h_3_3_cat_block = h_3_3
    h_3_4_cat_block = h_3_4

    h_3_1 = conv_bn_sigmoid(h_3_1, dim*4)
    h_3_2 = conv_bn_sigmoid(h_3_2, dim*4)
    h_3_3 = conv_bn_sigmoid(h_3_3, dim*4)
    h_3_4 = conv_bn_sigmoid(h_3_4, dim*4)

    h_3_1 = h_3[:, 0:input_shape[0] // 8, 0:input_shape[1] // 8, :] * h_3_1
    h_3_2 = h_3[:, 0:input_shape[0] // 8, input_shape[1] // 8:, :] * h_3_2
    h_3_3 = h_3[:, input_shape[0] // 8:, 0:input_shape[1] // 8, :] * h_3_3
    h_3_4 = h_3[:, input_shape[0] // 8:, input_shape[1] // 8:, :] * h_3_4
    
    h_3_12 = tf.concat([h_3_1, h_3_2], 2)
    h_3_34 = tf.concat([h_3_3, h_3_4], 2)
    h_3 = tf.concat([h_3_12, h_3_34], 1)

    h_4 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_3)
    h_4 = tf.keras.layers.Conv2D(filters=dim*8, kernel_size=3, padding="same", use_bias=False, name="block4_conv1")(h_4)
    h_4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)
    h_4 = tf.keras.layers.Conv2D(filters=dim*8, kernel_size=3, padding="same", use_bias=False, name="block4_conv2")(h_4)
    h_4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)
    h_4 = tf.keras.layers.Conv2D(filters=dim*8, kernel_size=3, padding="same", use_bias=False, name="block4_conv3")(h_4)
    h_4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)
    h_4_cat_block = h_4

    h_4_1 = h_4[:, 0:input_shape[0] // 16, 0:input_shape[1] // 16, :]
    h_4_2 = h_4[:, 0:input_shape[0] // 16, input_shape[1] // 16:, :]
    h_4_3 = h_4[:, input_shape[0] // 16:, 0:input_shape[1] // 16, :]
    h_4_4 = h_4[:, input_shape[0] // 16:, input_shape[1] // 16:, :]

    h_4_1 = res_connection(h_4_1, dim*8, True)
    h_4_2 = res_connection(h_4_2, dim*8, True)
    h_4_3 = res_connection(h_4_3, dim*8, True)
    h_4_4 = res_connection(h_4_4, dim*8, True)
    for _ in range(3):
        h_4_1 = res_connection(h_4_1, dim*8, False)
        h_4_2 = res_connection(h_4_2, dim*8, False)
        h_4_3 = res_connection(h_4_3, dim*8, False)
        h_4_4 = res_connection(h_4_4, dim*8, False)
    h_4_1_cat_block = h_4_1
    h_4_2_cat_block = h_4_2
    h_4_3_cat_block = h_4_3
    h_4_4_cat_block = h_4_4

    h_4_1 = conv_bn_sigmoid(h_4_1, dim*8)
    h_4_2 = conv_bn_sigmoid(h_4_2, dim*8)
    h_4_3 = conv_bn_sigmoid(h_4_3, dim*8)
    h_4_4 = conv_bn_sigmoid(h_4_4, dim*8)

    h_4_1 = h_4[:, 0:input_shape[0] // 16, 0:input_shape[1] // 16, :] * h_4_1
    h_4_2 = h_4[:, 0:input_shape[0] // 16, input_shape[1] // 16:, :] * h_4_2
    h_4_3 = h_4[:, input_shape[0] // 16:, 0:input_shape[1] // 16, :] * h_4_3
    h_4_4 = h_4[:, input_shape[0] // 16:, input_shape[1] // 16:, :] * h_4_4
    
    h_4_12 = tf.concat([h_4_1, h_4_2], 2)
    h_4_34 = tf.concat([h_4_3, h_4_4], 2)
    h_4 = tf.concat([h_4_12, h_4_34], 1)

    h_5 = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2)(h_4)
    h_5 = tf.keras.layers.Conv2D(filters=dim*8, kernel_size=3, padding="same", use_bias=False, name="block5_conv1")(h_5)
    h_5 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_5)
    h_5 = tf.keras.layers.ReLU()(h_5)
    h_5 = tf.keras.layers.Conv2D(filters=dim*8, kernel_size=3, padding="same", use_bias=False, name="block5_conv2")(h_5)
    h_5 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_5)
    h_5 = tf.keras.layers.ReLU()(h_5)
    h_5 = tf.keras.layers.Conv2D(filters=dim*8, kernel_size=3, padding="same", use_bias=False, name="block5_conv3")(h_5)
    h_5 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_5)
    h_5 = tf.keras.layers.ReLU()(h_5)

    h_5_1 = h_5[:, 0:input_shape[0] // 32, 0:input_shape[1] // 32, :]
    h_5_2 = h_5[:, 0:input_shape[0] // 32, input_shape[1] // 32:, :]
    h_5_3 = h_5[:, input_shape[0] // 32:, 0:input_shape[1] // 32, :]
    h_5_4 = h_5[:, input_shape[0] // 32:, input_shape[1] // 32:, :]
    h_5_1 = tf.keras.layers.Conv2DTranspose(filters=dim*4, kernel_size=2, strides=2, use_bias=False)(h_5_1)
    h_5_1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_5_1)
    h_5_1 = tf.keras.layers.ReLU()(h_5_1)
    h_5_2 = tf.keras.layers.Conv2DTranspose(filters=dim*4, kernel_size=2, strides=2, use_bias=False)(h_5_2)
    h_5_2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_5_2)
    h_5_2 = tf.keras.layers.ReLU()(h_5_2)
    h_5_3 = tf.keras.layers.Conv2DTranspose(filters=dim*4, kernel_size=2, strides=2, use_bias=False)(h_5_3)
    h_5_3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_5_3)
    h_5_3 = tf.keras.layers.ReLU()(h_5_3)
    h_5_4 = tf.keras.layers.Conv2DTranspose(filters=dim*4, kernel_size=2, strides=2, use_bias=False)(h_5_4)
    h_5_4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(h_5_4)
    h_5_4 = tf.keras.layers.ReLU()(h_5_4)

    hd4 = tf.keras.layers.Conv2DTranspose(filters=dim*4, kernel_size=2, strides=2, use_bias=False)(h_5)
    hd4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd4)
    hd4 = tf.keras.layers.ReLU()(hd4)
    hd4 = tf.concat([h_4_cat_block, hd4], -1)
    hd4 = tf.keras.layers.Conv2D(filters=dim*4, kernel_size=3, padding="same", use_bias=False)(hd4)
    hd4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd4)
    hd4 = tf.keras.layers.ReLU()(hd4)

    h_5_1 = decoder_res_connection(h_5_1, dim*4, True)
    h_5_2 = decoder_res_connection(h_5_2, dim*4, True)
    h_5_3 = decoder_res_connection(h_5_3, dim*4, True)
    h_5_4 = decoder_res_connection(h_5_4, dim*4, True)
    for _ in range(3):
        h_5_1 = decoder_res_connection(h_5_1, dim*4, False)
        h_5_2 = decoder_res_connection(h_5_2, dim*4, False)
        h_5_3 = decoder_res_connection(h_5_3, dim*4, False)
        h_5_4 = decoder_res_connection(h_5_4, dim*4, False)
    h_5_1 = tf.concat([h_4_1_cat_block, h_5_1], -1)
    h_5_2 = tf.concat([h_4_2_cat_block, h_5_2], -1)
    h_5_3 = tf.concat([h_4_3_cat_block, h_5_3], -1)
    h_5_4 = tf.concat([h_4_4_cat_block, h_5_4], -1)

    h_5_1 = conv_bn_sigmoid(h_5_1, dim*4)
    h_5_2 = conv_bn_sigmoid(h_5_2, dim*4)
    h_5_3 = conv_bn_sigmoid(h_5_3, dim*4)
    h_5_4 = conv_bn_sigmoid(h_5_4, dim*4)

    h_5_1 = hd4[:, 0:input_shape[0] // 16, 0:input_shape[1] // 16, :] * h_5_1
    h_5_2 = hd4[:, 0:input_shape[0] // 16, input_shape[1] // 16:, :] * h_5_2
    h_5_3 = hd4[:, input_shape[0] // 16:, 0:input_shape[1] // 16, :] * h_5_3
    h_5_4 = hd4[:, input_shape[0] // 16:, input_shape[1] // 16:, :] * h_5_4

    h_5_12 = tf.concat([h_5_1, h_5_2], 2)
    h_5_34 = tf.concat([h_5_3, h_5_4], 2)
    hd4 = tf.concat([h_5_12, h_5_34], 1)

    hd3_1 = hd4[:, 0:input_shape[0] // 16, 0:input_shape[1] // 16, :]
    hd3_2 = hd4[:, 0:input_shape[0] // 16, input_shape[1] // 16:, :]
    hd3_3 = hd4[:, input_shape[0] // 16:, 0:input_shape[1] // 16, :]
    hd3_4 = hd4[:, input_shape[0] // 16:, input_shape[1] // 16:, :]
    hd3_1 = tf.keras.layers.Conv2DTranspose(filters=dim*2, kernel_size=2, strides=2, use_bias=False)(hd3_1)
    hd3_1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd3_1)
    hd3_1 = tf.keras.layers.ReLU()(hd3_1)
    hd3_2 = tf.keras.layers.Conv2DTranspose(filters=dim*2, kernel_size=2, strides=2, use_bias=False)(hd3_2)
    hd3_2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd3_2)
    hd3_2 = tf.keras.layers.ReLU()(hd3_2)
    hd3_3 = tf.keras.layers.Conv2DTranspose(filters=dim*2, kernel_size=2, strides=2, use_bias=False)(hd3_3)
    hd3_3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd3_3)
    hd3_3 = tf.keras.layers.ReLU()(hd3_3)
    hd3_4 = tf.keras.layers.Conv2DTranspose(filters=dim*2, kernel_size=2, strides=2, use_bias=False)(hd3_4)
    hd3_4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd3_4)
    hd3_4 = tf.keras.layers.ReLU()(hd3_4)

    hd3 = tf.keras.layers.Conv2DTranspose(filters=dim*2, kernel_size=2, strides=2, use_bias=False)(hd4)
    hd3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd3)
    hd3 = tf.keras.layers.ReLU()(hd3)
    hd3 = tf.concat([h_3_cat_block, hd3], -1)
    hd3 = tf.keras.layers.Conv2D(filters=dim*2, kernel_size=3, padding="same", use_bias=False)(hd3)
    hd3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd3)
    hd3 = tf.keras.layers.ReLU()(hd3)

    hd3_1 = decoder_res_connection(hd3_1, dim*2, True)
    hd3_2 = decoder_res_connection(hd3_2, dim*2, True)
    hd3_3 = decoder_res_connection(hd3_3, dim*2, True)
    hd3_4 = decoder_res_connection(hd3_4, dim*2, True)
    for _ in range(3):
        hd3_1 = decoder_res_connection(hd3_1, dim*2, False)
        hd3_2 = decoder_res_connection(hd3_2, dim*2, False)
        hd3_3 = decoder_res_connection(hd3_3, dim*2, False)
        hd3_4 = decoder_res_connection(hd3_4, dim*2, False)
    hd3_1 = tf.concat([h_3_1_cat_block, hd3_1], -1)
    hd3_2 = tf.concat([h_3_2_cat_block, hd3_2], -1)
    hd3_3 = tf.concat([h_3_3_cat_block, hd3_3], -1)
    hd3_4 = tf.concat([h_3_4_cat_block, hd3_4], -1)

    hd3_1 = conv_bn_sigmoid(hd3_1, dim*2)
    hd3_2 = conv_bn_sigmoid(hd3_2, dim*2)
    hd3_3 = conv_bn_sigmoid(hd3_3, dim*2)
    hd3_4 = conv_bn_sigmoid(hd3_4, dim*2)

    hd3_1 = hd3[:, 0:input_shape[0] // 8, 0:input_shape[1] // 8, :] * hd3_1
    hd3_2 = hd3[:, 0:input_shape[0] // 8, input_shape[1] // 8:, :] * hd3_2
    hd3_3 = hd3[:, input_shape[0] // 8:, 0:input_shape[1] // 8, :] * hd3_3
    hd3_4 = hd3[:, input_shape[0] // 8:, input_shape[1] // 8:, :] * hd3_4

    hd3_12 = tf.concat([hd3_1, hd3_2], 2)
    hd3_34 = tf.concat([hd3_3, hd3_4], 2)
    hd3 = tf.concat([hd3_12, hd3_34], 1)

    hd2_1 = hd3[:, 0:input_shape[0] // 8, 0:input_shape[1] // 8 :]
    hd2_2 = hd3[:, 0:input_shape[0] // 8, input_shape[1] // 8:, :]
    hd2_3 = hd3[:, input_shape[0] // 8:, 0:input_shape[1] // 8, :]
    hd2_4 = hd3[:, input_shape[0] // 8:, input_shape[1] // 8:, :]
    hd2_1 = tf.keras.layers.Conv2DTranspose(filters=dim, kernel_size=2, strides=2, use_bias=False)(hd2_1)
    hd2_1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd2_1)
    hd2_1 = tf.keras.layers.ReLU()(hd2_1)
    hd2_2 = tf.keras.layers.Conv2DTranspose(filters=dim, kernel_size=2, strides=2, use_bias=False)(hd2_2)
    hd2_2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd2_2)
    hd2_2 = tf.keras.layers.ReLU()(hd2_2)
    hd2_3 = tf.keras.layers.Conv2DTranspose(filters=dim, kernel_size=2, strides=2, use_bias=False)(hd2_3)
    hd2_3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd2_3)
    hd2_3 = tf.keras.layers.ReLU()(hd2_3)
    hd2_4 = tf.keras.layers.Conv2DTranspose(filters=dim, kernel_size=2, strides=2, use_bias=False)(hd2_4)
    hd2_4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd2_4)
    hd2_4 = tf.keras.layers.ReLU()(hd2_4)

    hd2 = tf.keras.layers.Conv2DTranspose(filters=dim, kernel_size=2, strides=2, use_bias=False)(hd3)
    hd2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd2)
    hd2 = tf.keras.layers.ReLU()(hd2)
    hd2 = tf.concat([h_2_cat_block, hd2], -1)
    hd2 = tf.keras.layers.Conv2D(filters=dim, kernel_size=3, padding="same", use_bias=False)(hd2)
    hd2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd2)
    hd2 = tf.keras.layers.ReLU()(hd2)

    hd2_1 = decoder_res_connection(hd2_1, dim, True)
    hd2_2 = decoder_res_connection(hd2_2, dim, True)
    hd2_3 = decoder_res_connection(hd2_3, dim, True)
    hd2_4 = decoder_res_connection(hd2_4, dim, True)
    for _ in range(2):
        hd2_1 = decoder_res_connection(hd2_1, dim, False)
        hd2_2 = decoder_res_connection(hd2_2, dim, False)
        hd2_3 = decoder_res_connection(hd2_3, dim, False)
        hd2_4 = decoder_res_connection(hd2_4, dim, False)
    hd2_1 = tf.concat([h_2_1_cat_block, hd2_1], -1)
    hd2_2 = tf.concat([h_2_2_cat_block, hd2_2], -1)
    hd2_3 = tf.concat([h_2_3_cat_block, hd2_3], -1)
    hd2_4 = tf.concat([h_2_4_cat_block, hd2_4], -1)

    hd2_1 = conv_bn_sigmoid(hd2_1, dim)
    hd2_2 = conv_bn_sigmoid(hd2_2, dim)
    hd2_3 = conv_bn_sigmoid(hd2_3, dim)
    hd2_4 = conv_bn_sigmoid(hd2_4, dim)

    hd2_1 = hd2[:, 0:input_shape[0] // 4, 0:input_shape[1] // 4, :] * hd2_1
    hd2_2 = hd2[:, 0:input_shape[0] // 4, input_shape[1] // 4:, :] * hd2_2
    hd2_3 = hd2[:, input_shape[0] // 4:, 0:input_shape[1] // 4, :] * hd2_3
    hd2_4 = hd2[:, input_shape[0] // 4:, input_shape[1] // 4:, :] * hd2_4

    hd2_12 = tf.concat([hd2_1, hd2_2], 2)
    hd2_34 = tf.concat([hd2_3, hd2_4], 2)
    hd2 = tf.concat([hd2_12, hd2_34], 1)

    hd1_1 = hd2[:, 0:input_shape[0] // 4, 0:input_shape[1] // 4, :]
    hd1_2 = hd2[:, 0:input_shape[0] // 4, input_shape[1] // 4:, :]
    hd1_3 = hd2[:, input_shape[0] // 4:, 0:input_shape[1] // 4, :]
    hd1_4 = hd2[:, input_shape[0] // 4:, input_shape[1] // 4:, :]

    hd1_1 = tf.keras.layers.Conv2DTranspose(filters=dim // 2, kernel_size=2, strides=2, use_bias=False)(hd1_1)
    hd1_1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd1_1)
    hd1_1 = tf.keras.layers.ReLU()(hd1_1)
    hd1_2 = tf.keras.layers.Conv2DTranspose(filters=dim // 2, kernel_size=2, strides=2, use_bias=False)(hd1_2)
    hd1_2 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd1_2)
    hd1_2 = tf.keras.layers.ReLU()(hd1_2)
    hd1_3 = tf.keras.layers.Conv2DTranspose(filters=dim // 2, kernel_size=2, strides=2, use_bias=False)(hd1_3)
    hd1_3 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd1_3)
    hd1_3 = tf.keras.layers.ReLU()(hd1_3)
    hd1_4 = tf.keras.layers.Conv2DTranspose(filters=dim // 2, kernel_size=2, strides=2, use_bias=False)(hd1_4)
    hd1_4 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd1_4)
    hd1_4 = tf.keras.layers.ReLU()(hd1_4)

    hd1 = tf.keras.layers.Conv2DTranspose(filters=dim // 2, kernel_size=2, strides=2, use_bias=False)(hd2)
    hd1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd1)
    hd1 = tf.keras.layers.ReLU()(hd1)
    hd1 = tf.concat([h_1_cat_block, hd1], -1)
    hd1 = tf.keras.layers.Conv2D(filters=dim // 2, kernel_size=3, padding="same", use_bias=False)(hd1)
    hd1 = tf.keras.layers.BatchNormalization(epsilon=1.001e-5)(hd1)
    hd1 = tf.keras.layers.ReLU()(hd1)

    hd1_1 = decoder_res_connection(hd1_1, dim // 2, True)
    hd1_2 = decoder_res_connection(hd1_2, dim // 2, True)
    hd1_3 = decoder_res_connection(hd1_3, dim // 2, True)
    hd1_4 = decoder_res_connection(hd1_4, dim // 2, True)
    for _ in range(2):
        hd1_1 = decoder_res_connection(hd1_1, dim // 2, False)
        hd1_2 = decoder_res_connection(hd1_2, dim // 2, False)
        hd1_3 = decoder_res_connection(hd1_3, dim // 2, False)
        hd1_4 = decoder_res_connection(hd1_4, dim // 2, False)
    hd1_1 = tf.concat([h_1_1_cat_block, hd1_1], -1)
    hd1_2 = tf.concat([h_1_2_cat_block, hd1_2], -1)
    hd1_3 = tf.concat([h_1_3_cat_block, hd1_3], -1)
    hd1_4 = tf.concat([h_1_4_cat_block, hd1_4], -1)

    hd1_1 = conv_bn_sigmoid(hd1_1, dim // 2)
    hd1_2 = conv_bn_sigmoid(hd1_2, dim // 2)
    hd1_3 = conv_bn_sigmoid(hd1_3, dim // 2)
    hd1_4 = conv_bn_sigmoid(hd1_4, dim // 2)

    hd1_1 = hd1[:, 0:input_shape[0] // 2, 0:input_shape[1] // 2, :] * hd1_1
    hd1_1 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1, name='final_patch_1')(hd1_1)
    hd1_2 = hd1[:, 0:input_shape[0] // 2, input_shape[1] // 2:, :] * hd1_2
    hd1_2 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1, name='final_patch_2')(hd1_2)
    hd1_3 = hd1[:, input_shape[0] // 2:, 0:input_shape[1] // 2, :] * hd1_3
    hd1_3 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1, name='final_patch_3')(hd1_3)
    hd1_4 = hd1[:, input_shape[0] // 2:, input_shape[1] // 2:, :] * hd1_4
    hd1_4 = tf.keras.layers.Conv2D(filters=nclasses, kernel_size=1, name='final_patch_4')(hd1_4)


    model = tf.keras.Model(inputs=inputs, outputs=[hd1_1, hd1_2, hd1_3, hd1_4])

    return model
