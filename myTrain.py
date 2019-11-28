import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
import math
import pickle
from utils import basic_hyperparams
from GeoMAN import GeoMAN
from utils import my_get_batch_feed_dict
from utils import shuffle_data
from utils import my_get_valid_batch_feed_dict


def train_HangZhou():
    np.random.seed(2017)

    # use specific gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    # load hyperparameters
    hps = basic_hyperparams()
    hps_dict = json.load(open('./hparam_files/AirQualityGeoMan.json', 'r'))
    hps.override_from_dict(hps_dict)
    local_inputs = np.load('./data/local_inputs_0_in.npy')  # shape=(n,5,2)
    global_inputs = np.load('./data/global_inputs_0_in.npy')  # shape=(n,5,81)
    global_attn_states = np.load('./data/global_attn_states.npy')  # shape=(n,81,2,5)
    external_inputs = np.load('./data/external_inputs.npy')  # shape=(n,1,5)
    labels = np.load('./data/labels_0_in.npy')  # shape=(n,1,1)
    training_data = [local_inputs, global_inputs, global_attn_states, external_inputs, labels]

    val_local_inputs = np.load('./data/val_local_inputs_0_in.npy')
    val_global_inputs = np.load('./data/val_global_inputs_0_in.npy')
    val_global_attn_states = np.load('./data/val_global_attn_states.npy')
    val_external_inputs = np.load('./data/val_external_inputs.npy')
    val_labels = np.load('./data/val_labels_0_in.npy')
    valid_data = [val_local_inputs, val_global_inputs, val_global_attn_states, val_external_inputs, val_labels]
    num_train = len(training_data[0])
    num_valid = len(valid_data[0])

    # model construction
    tf.reset_default_graph()
    model = GeoMAN(hps)
    # print trainable params
    for i in tf.trainable_variables():
        print(i)
    # print all placeholders
    phs = [x for x in tf.get_default_graph().get_operations()
           if x.type == "Placeholder"]
    # count the parameters in our model
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('total parameters: {}'.format(total_parameters))

    # path for log saving
    if hps.ext_flag:
        if hps.s_attn_flag == 0:
            model_name = 'GeoMANng'
        elif hps.s_attn_flag == 1:
            model_name = 'GeoMANnl'
        else:
            model_name = 'GeoMAN'
    else:
        model_name = 'GeoMANne'
    logdir = './logs/{}-{}-{}-{}-{}-{:.2f}-{:.3f}/'.format(model_name,
                                                           hps.n_steps_encoder,
                                                           hps.n_steps_decoder,
                                                           hps.n_stacked_layers,
                                                           hps.n_hidden_encoder,
                                                           hps.dropout_rate,
                                                           hps.lambda_l2_reg)
    model_dir = logdir + 'saved_models/'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    results_dir = logdir + 'results/'

    # train params
    total_epoch = 100
    batch_size = 64
    display_iter = 1
    save_log_iter = 10
    n_split_valid = 100  # times of splitting validation set
    valid_losses = [np.inf]

    # training process
    with tf.Session() as sess:
        saver = tf.train.Saver()

        # initialize
        model.init(sess)
        iter = 0
        summary_writer = tf.summary.FileWriter(logdir)

        for i in range(total_epoch):
            print('----------epoch {}-----------'.format(i))
            training_data = shuffle_data(training_data)

            for j in range(0, num_train, batch_size):
                iter += 1
                feed_dict = my_get_batch_feed_dict(model, j, batch_size, training_data)
                _, merged_summary = sess.run(
                    [model.phs['train_op'], model.phs['summary']], feed_dict)
                # summary_writer.add_summary(merged_summary, iter)
                if iter % save_log_iter == 0:
                    summary_writer.add_summary(merged_summary, iter)
                if iter % display_iter == 0:
                    valid_loss = 0
                    valid_indexes = np.int64(
                        np.linspace(0, num_valid, n_split_valid))
                    for k in range(n_split_valid - 1):
                        feed_dict = my_get_valid_batch_feed_dict(model, valid_indexes, k, valid_data)
                        batch_loss = sess.run(model.phs['loss'], feed_dict)
                        valid_loss += batch_loss
                    valid_loss /= n_split_valid - 1
                    valid_losses.append(valid_loss)
                    valid_loss_sum = tf.Summary(
                        value=[tf.Summary.Value(tag="valid_loss", simple_value=valid_loss)])
                    summary_writer.add_summary(valid_loss_sum, iter)

                    if valid_loss < min(valid_losses[:-1]):
                        print('iter {}\tvalid_loss = {:.6f}\tmodel saved!!'.format(
                            iter, valid_loss))
                        saver.save(sess, model_dir +
                                   'model_{}.ckpt'.format(iter))
                        saver.save(sess, model_dir + 'final_model.ckpt')
                    else:
                        print('iter {}\tvalid_loss = {:.6f}\t'.format(
                            iter, valid_loss))

    print('stop training !!!')


def train_TaxiBJ():
    np.random.seed(2017)

    # use specific gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)

    # load hyperparameters
    hps = basic_hyperparams()
    hps_dict = json.load(open('./hparam_files/TaxiBjGeoMan.json', 'r'))
    hps.override_from_dict(hps_dict)
    local_inputs = np.load('./data/local_inputs_0_in_TaxiBJ.npy')  # shape=(n,5,2)
    global_inputs = np.load('./data/global_inputs_0_in_TaxiBJ.npy')  # shape=(n,5,81)
    global_attn_states = np.load('./data/global_attn_states_TaxiBJ.npy')  # shape=(n,81,2,5)
    external_inputs = np.load('./data/external_inputs_TaxiBJ.npy')  # shape=(n,1,5)
    labels = np.load('./data/labels_0_in_TaxiBJ.npy')  # shape=(n,1,1)
    training_data = [local_inputs, global_inputs, global_attn_states, external_inputs, labels]

    val_local_inputs = np.load('./data/val_local_inputs_0_in_taxibj_0404.npy')
    val_global_inputs = np.load('./data/val_global_inputs_0_in_taxibj_0404.npy')
    val_global_attn_states = np.load('./data/val_global_attn_states_taxibj_0404.npy')
    val_external_inputs = np.load('./data/val_external_inputs_taxibj.npy')
    val_labels = np.load('./data/val_labels_0_in_taxibj_0404.npy')
    valid_data = [val_local_inputs, val_global_inputs, val_global_attn_states, val_external_inputs, val_labels]
    num_train = len(training_data[0])
    num_valid = len(valid_data[0])

    # model construction
    tf.reset_default_graph()
    model = GeoMAN(hps)
    # print trainable params
    for i in tf.trainable_variables():
        print(i)
    # print all placeholders
    phs = [x for x in tf.get_default_graph().get_operations()
           if x.type == "Placeholder"]
    # count the parameters in our model
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('total parameters: {}'.format(total_parameters))

    # path for log saving
    if hps.ext_flag:
        if hps.s_attn_flag == 0:
            model_name = 'GeoMANng'
        elif hps.s_attn_flag == 1:
            model_name = 'GeoMANnl'
        else:
            model_name = 'GeoMAN_TaxiBJ'
    else:
        model_name = 'GeoMANne'
    logdir = './logs/{}-{}-{}-{}-{}-{:.2f}-{:.3f}/'.format(model_name,
                                                           hps.n_steps_encoder,
                                                           hps.n_steps_decoder,
                                                           hps.n_stacked_layers,
                                                           hps.n_hidden_encoder,
                                                           hps.dropout_rate,
                                                           hps.lambda_l2_reg)
    model_dir = logdir + 'saved_models/'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    results_dir = logdir + 'results/'

    # train params
    total_epoch = 50
    batch_size = 32
    display_iter = 1
    save_log_iter = 10
    n_split_valid = 10  # times of splitting validation set
    valid_losses = [np.inf]

    # training process
    with tf.Session() as sess:
        saver = tf.train.Saver()

        # initialize
        model.init(sess)
        iter = 0
        summary_writer = tf.summary.FileWriter(logdir)

        for i in range(total_epoch):
            print('----------epoch {}-----------'.format(i))
            training_data = shuffle_data(training_data)

            for j in range(0, num_train, batch_size):
                iter += 1
                feed_dict = my_get_batch_feed_dict(model, j, batch_size, training_data)
                _, merged_summary = sess.run(
                    [model.phs['train_op'], model.phs['summary']], feed_dict)
                # summary_writer.add_summary(merged_summary, iter)
                if iter % save_log_iter == 0:
                    summary_writer.add_summary(merged_summary, iter)
                if iter % display_iter == 0:
                    valid_loss = 0
                    valid_indexes = np.int64(
                        np.linspace(0, num_valid, n_split_valid))
                    for k in range(n_split_valid - 1):
                        feed_dict = my_get_valid_batch_feed_dict(model, valid_indexes, k, valid_data)
                        batch_loss = sess.run(model.phs['loss'], feed_dict)
                        valid_loss += batch_loss
                    valid_loss /= n_split_valid - 1
                    valid_losses.append(valid_loss)
                    valid_loss_sum = tf.Summary(
                        value=[tf.Summary.Value(tag="valid_loss", simple_value=valid_loss)])
                    summary_writer.add_summary(valid_loss_sum, iter)

                    if valid_loss < min(valid_losses[:-1]):
                        print('iter {}\tvalid_loss = {:.6f}\tmodel saved!!'.format(
                            iter, valid_loss))
                        saver.save(sess, model_dir +
                                   'model_{}.ckpt'.format(iter))
                        saver.save(sess, model_dir + 'final_model.ckpt')
                    else:
                        print('iter {}\tvalid_loss = {:.6f}\t'.format(
                            iter, valid_loss))

    print('stop training !!!')


if __name__ == '__main__':
    train_TaxiBJ()
