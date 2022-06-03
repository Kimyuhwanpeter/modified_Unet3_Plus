# -*- coding:utf-8 -*-
from model_15 import *
from random import random, shuffle
from tensorflow.keras import backend as K
from Cal_measurement import *

import matplotlib.pyplot as plt
import numpy as np
import easydict
import os

FLAGS = easydict.EasyDict({"img_size": 448,

                           "train_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/Apple_A/train_fix.txt",

                           "test_txt_path": "/yuhwan/yuhwan/Dataset/Segmentation/Apple_A/test.txt",
                           
                           "tr_label_path": "/yuhwan/yuhwan/Dataset/Segmentation/Apple_A/aug_train_lab/",
                           
                           "tr_image_path": "/yuhwan/yuhwan/Dataset/Segmentation/Apple_A/aug_train_img/",

                           "te_label_path": "/yuhwan/yuhwan/Dataset/Segmentation/Apple_A/All_labels/",
                           
                           "te_image_path": "/yuhwan/yuhwan/Dataset/Segmentation/Apple_A/All_images/",
                           
                           "pre_checkpoint": False,
                           
                           "pre_checkpoint_path": "C:/Users/Yuhwan/Downloads/398/398",
                           
                           "lr": 3e-4,

                           "min_lr": 1e-7,
                           
                           "epochs": 400,

                           "total_classes": 3,

                           "ignore_label": 0,

                           "batch_size": 2,

                           "sample_images": "/yuhwan/Edisk/yuhwan/Edisk/Segmentation/6th_paper/proposed_method/Apple_A/sample_images",

                           "save_checkpoint": "/yuhwan/Edisk/yuhwan/Edisk/Segmentation/6th_paper/proposed_method/Apple_A/checkpoint",

                           "save_print": "/yuhwan/Edisk/yuhwan/Edisk/Segmentation/6th_paper/proposed_method/Apple_A/train_out.txt",

                           "train_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_loss.txt",

                           "train_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/train_acc.txt",

                           "val_loss_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_loss.txt",

                           "val_acc_graphs": "/yuwhan/Edisk/yuwhan/Edisk/Segmentation/V2/BoniRob/val_acc.txt",

                           "test_images": "C:/Users/Yuhwan/Downloads/test_images",

                           "train": True})

optim = tf.keras.optimizers.Adam(FLAGS.lr)
color_map = np.array([[0, 0, 0],[255,0,0]], np.uint8)

def tr_func(image_list, label_list):

    h = tf.random.uniform([1], 1e-2, 30)
    h = tf.cast(tf.math.ceil(h[0]), tf.int32)
    w = tf.random.uniform([1], 1e-2, 30)
    w = tf.cast(tf.math.ceil(w[0]), tf.int32)

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.cast(img, tf.float32)
    img = tf.image.random_brightness(img, max_delta=50.) 
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_hue(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    # img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, 3], seed=123)
    no_img = img
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_jpeg(lab, 1)
    lab = tf.image.resize(lab, [FLAGS.img_size, FLAGS.img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
     #lab = tf.image.random_crop(lab, [FLAGS.img_size, FLAGS.img_size, 1], seed=123)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    if random() > 0.5:
        img = tf.image.flip_left_right(img)
        lab = tf.image.flip_left_right(lab)
        
    return img, no_img, lab

def test_func(image_list, label_list):

    img = tf.io.read_file(image_list)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.resize(img, [1024, 1024])
    #img = tf.clip_by_value(img, 0, 255)
    # img = img[:, :, ::-1] - tf.constant([103.939, 116.779, 123.68])
    img = img / 255.

    lab = tf.io.read_file(label_list)
    lab = tf.image.decode_jpeg(lab, 1)
    lab = tf.image.resize(lab, [1024, 1024], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    lab = tf.image.convert_image_dtype(lab, tf.uint8)

    return img, lab

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -tf.keras.backend.mean((alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               + ((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0)))
        # return -tf.keras.backend.sum(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
        #        -tf.keras.backend.sum((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    return binary_focal_loss_fixed

def iou_loss_fn(y_true, y_pred):

    y_true = tf.reshape(y_true, [-1,])
    y_pred = tf.reshape(y_pred, [-1,])

    y_pred = tf.cast(tf.nn.sigmoid(y_pred), tf.float32)
    y_true = tf.cast(y_true, y_pred.dtype)

    area_intersect = tf.reduce_sum(tf.multiply(y_true, y_pred))

    area_true = tf.reduce_sum(y_true)
    area_pred = tf.reduce_sum(y_pred)
    area_union = area_true + area_pred - area_intersect

    return 1-tf.math.divide_no_nan(area_intersect, area_union)

def MS_SSIM_fn(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.where(y_pred >= 0.5, 1., 0.)
    y_pred = tf.cast(y_pred, tf.float32)
    
    tf_ms_ssim = tf.image.ssim_multiscale(y_true, y_pred, max_val=1.0, k1=0.01**2, k2=0.03**2)
    
    loss = 1 - tf.reduce_mean(tf_ms_ssim)
    
    return loss

def hybrid_loss(y_true, y_pred, alpha):

    focal_loss = binary_focal_loss(alpha=alpha)(tf.reshape(y_true, [-1,]), tf.nn.sigmoid(tf.reshape(y_pred, [-1,])))  # for CGM
    iou_loss = iou_loss_fn(y_true, y_pred)  # Saptial loss
    MS_SSIM_loss = MS_SSIM_fn(y_true, tf.nn.sigmoid(y_pred)) # enhance the boundary

    return focal_loss + iou_loss + MS_SSIM_loss


def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, batch_images, batch_labels, object_buf):

    with tf.GradientTape() as tape:
        final, sub_1, sub_2, sub_3, sub_4 = run_model(model, batch_images, True)

        loss_1 = hybrid_loss(batch_labels, sub_1, object_buf[1]) * 0.25
        loss_2 = hybrid_loss(batch_labels, sub_2, object_buf[1]) * 0.25
        loss_3 = hybrid_loss(batch_labels, sub_3, object_buf[1]) * 0.25
        loss_4 = hybrid_loss(batch_labels, sub_4, object_buf[1]) * 0.25
        loss_5 = hybrid_loss(batch_labels, final, object_buf[1]) * 1

        total_loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5

    grads = tape.gradient(total_loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return total_loss

def main():

    model = Unet3_plus_modified(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), nclasses=1)
    model_prob = model_profiler(model, FLAGS.batch_size)
    model.summary()
    print(model_prob)

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manager.latest_checkpoint:
            ckpt.restore(ckpt_manager.latest_checkpoint)

    if FLAGS.train:
        count = 0

        output_text = open(FLAGS.save_print, "w")
        
        train_list = np.loadtxt(FLAGS.train_txt_path, dtype="<U200", skiprows=0, usecols=0)
        test_list = np.loadtxt(FLAGS.test_txt_path, dtype="<U200", skiprows=0, usecols=0)

        train_img_dataset = [FLAGS.tr_image_path + data for data in train_list]
        test_img_dataset = [FLAGS.te_image_path + data for data in test_list]

        train_lab_dataset = [FLAGS.tr_label_path + data for data in train_list]
        test_lab_dataset = [FLAGS.te_label_path + data for data in test_list]

        for epoch in range(FLAGS.epochs):
            A = list(zip(train_img_dataset, train_lab_dataset))
            shuffle(A)
            train_img_dataset, train_lab_dataset = zip(*A)
            train_img_dataset, train_lab_dataset = np.array(train_img_dataset), np.array(train_lab_dataset)

            train_ge = tf.data.Dataset.from_tensor_slices((train_img_dataset, train_lab_dataset))
            train_ge = train_ge.shuffle(len(train_img_dataset))
            train_ge = train_ge.map(tr_func)
            train_ge = train_ge.batch(FLAGS.batch_size)
            train_ge = train_ge.prefetch(tf.data.experimental.AUTOTUNE)
            tr_iter = iter(train_ge)

            tr_idx = len(train_img_dataset) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                batch_labels = tf.where(batch_labels == 255, 1, batch_labels).numpy()

                class_imbal_labels_buf = 0.
                class_imbal_labels = batch_labels
                for i in range(FLAGS.batch_size):
                    class_imbal_label = class_imbal_labels[i]
                    class_imbal_label = np.reshape(class_imbal_label, [FLAGS.img_size*FLAGS.img_size, ])
                    count_c_i_lab = np.bincount(class_imbal_label, minlength=2)
                    class_imbal_labels_buf += count_c_i_lab

                object_buf = class_imbal_labels_buf
                object_buf = (np.max(object_buf / np.sum(object_buf)) + 1 - (object_buf / np.sum(object_buf)))
                object_buf = tf.nn.softmax(object_buf).numpy()
                
                loss = cal_loss(model, batch_images, batch_labels, object_buf)

                if count % 10 == 0:
                    print("Epochs: {}, Loss = {} [{}/{}]".format(epoch, loss, step + 1, tr_idx))

                if count % 100 == 0:
                    final, _, _, _, _ = run_model(model, batch_images, False)
                    for i in range(FLAGS.batch_size):
                        label = tf.cast(batch_labels[i, :, :, 0], tf.int32).numpy()

                        output_ = tf.where(tf.nn.sigmoid(final[i, :, :, 0]) >= 0.5, 1, 0)

                        output_ = tf.cast(output_, tf.int32)
                        final_output = output_

                        pred_mask_color = color_map[final_output]
                        label_mask_color = color_map[label]
 
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_label.png", label_mask_color)
                        plt.imsave(FLAGS.sample_images + "/{}_batch_{}".format(count, i) + "_predict.png", pred_mask_color)

                count += 1

            tr_iter = iter(train_ge)
            iou = 0.
            cm = 0.
            f1_score_ = 0.
            recall_ = 0.
            precision_ = 0.
            for i in range(tr_idx):
                batch_images, _, batch_labels = next(tr_iter)
                batch_labels = tf.where(batch_labels == 255, 1, batch_labels).numpy()
                for j in range(FLAGS.batch_size):
                    batch_image = tf.expand_dims(batch_images[j], 0)
                    final, _, _, _, _ = run_model(model, batch_image, False)

                    output_ = tf.where(tf.nn.sigmoid(final[0, :, :, 0]) >= 0.5, 1, 0)
 
                    output_ = tf.cast(output_, tf.int32)
                    final_output = output_
                    final_output = tf.where(final_output == 0, 1, 0)

                    batch_label = tf.cast(batch_labels[j, :, :, 0], tf.uint8).numpy()
                    batch_label = np.where(batch_label == 0, 1, 0)
                    batch_label = np.array(batch_label, np.int32)

                    cm_ = Measurement(predict=final_output,
                                        label=batch_label, 
                                        shape=[FLAGS.img_size*FLAGS.img_size, ], 
                                        total_classes=2).MIOU()
                    
                    cm += cm_

                iou = cm[0,0]/(cm[0,0] + cm[0,1] + cm[1,0])
                precision_ = cm[0,0] / (cm[0,0] + cm[1,0])
                recall_ = cm[0,0] / (cm[0,0] + cm[0,1])
                f1_score_ = (2*precision_*recall_) / (precision_ + recall_)
            print("train mIoU = %.4f, train F1_score = %.4f, train sensitivity(recall) = %.4f, train precision = %.4f" % (iou,
                                                                                                                        f1_score_,
                                                                                                                        recall_,
                                                                                                                        precision_))

            output_text.write("Epoch: ")
            output_text.write(str(epoch))
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.write("train IoU: ")
            output_text.write("%.4f" % (iou ))
            output_text.write(", train F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", train sensitivity: ")
            output_text.write("%.4f" % (recall_ ))
            output_text.write(", train precision: ")
            output_text.write("%.4f" % (precision_ ))
            output_text.write("\n")
        
            test_ge = tf.data.Dataset.from_tensor_slices((test_img_dataset, test_lab_dataset))
            test_ge = test_ge.map(test_func)
            test_ge = test_ge.batch(1)
            test_ge = test_ge.prefetch(tf.data.experimental.AUTOTUNE)

            test_iter = iter(test_ge)
            iou = 0.
            cm = 0.
            f1_score_ = 0.
            recall_ = 0.
            precision_ = 0.
            # model_ = multi_region_class(input_shape=(FLAGS.img_size*2, FLAGS.img_size*2, 3), nclasses=2)
            # model_.set_weights(model.get_weights())
            for i in range(len(test_img_dataset)):
                batch_images, batch_labels = next(test_iter)
                batch_labels = tf.where(batch_labels == 255, 1, batch_labels).numpy()

                image = batch_images[0].numpy()
                # shape = tf.keras.backend.int_shape(batch_labels[0].numpy())
                height = batch_labels.shape[1]
                width = batch_labels.shape[2]
                final, _, _, _, _ = run_model(model, batch_images, False)
                output_ = tf.where(tf.nn.sigmoid(final[0, :, :, 0]) >= 0.5, 1, 0)

                output_ = tf.cast(output_, tf.int32)
                final_output = output_
                final_output = tf.where(final_output == 0, 1, 0)
                #final_output = tf.expand_dims(output, -1)
                #final_output = tf.image.resize(final_output, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                #final_output = final_output[:, :, 0]

                # desired_h = 1024    # Apple A   if height == 3456 and width == 5184:    # Apple A
                # desired_w = 1024    # Apple A   if height == 3456 and width == 5184:    # Apple A
                # for j in range(2):
                #     for k in range(2):
                #         split_img = image[j*desired_h:(j+1)*desired_h, k*desired_w:(k+1)*desired_w, :]
                #         split_img = tf.expand_dims(split_img, 0)
                #         # split_img = tf.image.resize(split_img, [FLAGS.img_size, FLAGS.img_size])
                #         second_output = run_model(model_, split_img, False)
                #         output = tf.nn.softmax(second_output[0, :, :, :], -1)
                #         output = tf.argmax(output, -1, output_type=tf.int32)
                #         output = tf.where(output == 0, 1, 0)
                #         #temp_image = tf.image.resize(split_img, [1728, 1728]).numpy()
                #         #temp_image = temp_image[0]

                #         if j == 0 and k == 0:
                #             final_output = output
                #             #final_image = temp_image
                #         else:
                #             if j == 0:
                #                 final_output = tf.concat([final_output, output], 1)
                #                 #final_image = tf.concat([final_image, temp_image], 1)
                #             else:
                #                 if j == 1 and k == 0:
                #                     final_output2 = output
                #                     #final_image2 = temp_image
                #                 else:
                #                     final_output2 = tf.concat([final_output2, output], 1)
                #                     #final_image2 = tf.concat([final_image2, temp_image], 1)

                # #final_image3 = tf.concat([final_image, final_image2], 0)
                # final_output = tf.concat([final_output, final_output2], 0)
                # final_output = tf.expand_dims(final_output, -1)
                # final_output = tf.image.resize(final_output, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                # final_output = final_output[:, :, 0]
                batch_label = tf.cast(batch_labels[0, :, :, 0], tf.uint8).numpy()
                batch_label = np.where(batch_label == 0, 1, 0)
                batch_label = np.array(batch_label, np.int32)

                cm_ = Measurement(predict=final_output,
                                    label=batch_label,
                                    shape=[height*width, ],
                                    total_classes=2).MIOU()
                    
                cm += cm_

                iou = cm[0,0]/(cm[0,0] + cm[0,1] + cm[1,0])
                precision_ = cm[0,0] / (cm[0,0] + cm[1,0])
                recall_ = cm[0,0] / (cm[0,0] + cm[0,1])
                f1_score_ = (2*precision_*recall_) / (precision_ + recall_)

            print("test mIoU = %.4f, test F1_score = %.4f, test sensitivity(recall) = %.4f, test precision = %.4f" % (iou,
                                                                                                                    f1_score_,
                                                                                                                    recall_,
                                                                                                                    precision_))
            output_text.write("test IoU: ")
            output_text.write("%.4f" % (iou))
            output_text.write(", test F1_score: ")
            output_text.write("%.4f" % (f1_score_))
            output_text.write(", test sensitivity: ")
            output_text.write("%.4f" % (recall_ ))
            output_text.write(", test precision: ")
            output_text.write("%.4f" % (precision_))
            output_text.write("\n")
            output_text.write("===================================================================")
            output_text.write("\n")
            output_text.flush()

            model_dir = "%s/%s" % (FLAGS.save_checkpoint, epoch)
            if not os.path.isdir(model_dir):
                print("Make {} folder to store the weight!".format(epoch))
                os.makedirs(model_dir)
            ckpt = tf.train.Checkpoint(model=model, optim=optim)
            ckpt_dir = model_dir + "/apple_model_{}.ckpt".format(epoch)
            ckpt.save(ckpt_dir)



if __name__ == "__main__":
    main()
