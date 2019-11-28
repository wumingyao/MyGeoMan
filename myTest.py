import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import json
import math
import pickle
from utils import basic_hyperparams
from utils import load_data
from utils import load_global_inputs
from utils import my_get_valid_batch_feed_dict
from errors import *


def root_mean_squared_error(labels, preds):
    total_size = np.size(labels)
    return np.sqrt(np.sum(np.square(labels - preds)) / total_size)


def mean_absolute_error(labels, preds):
    total_size = np.size(labels)
    return np.sum(np.abs(labels - preds)) / total_size


def test_HangZhou():
    # use specific gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # load hyperparameters
    session = tf.Session(config=tf_config)
    hps = basic_hyperparams()
    hps_dict = json.load(open('./hparam_files/AirQualityGeoMan.json', 'r'))
    hps.override_from_dict(hps_dict)

    # model construction
    tf.reset_default_graph()
    from GeoMAN import GeoMAN

    print(hps)
    model = GeoMAN(hps)

    # read data from test set
    input_path = './data/'
    val_local_inputs = np.load('./data/test_local_inputs_0_in_day28.npy')
    val_global_inputs = np.load('./data/test_global_inputs_0_in_day28.npy')
    val_global_attn_states = np.load('./data/test_global_attn_states_day28.npy')
    val_external_inputs = np.load('./data/val_external_inputs.npy')
    val_labels = np.load('./data/test_labels_0_in_day28.npy')
    test_data = [val_local_inputs, val_global_inputs, val_global_attn_states, val_external_inputs, val_labels]
    num_test = len(test_data[0])
    print('test samples: {0}'.format(num_test))

    # read scaler of the labels
    f = open('./data/scalers/scaler-0.pkl', 'rb')
    scaler = pickle.load(f)
    f.close()

    # path
    if hps.ext_flag:
        if hps.s_attn_flag == 0:
            model_name = 'GeoMANng'
        elif hps.s_attn_flag == 1:
            model_name = 'GeoMANnl'
        else:
            model_name = 'GeoMAN'
    else:
        model_name = 'GeoMANne'
    model_path = './logs/{}-{}-{}-{}-{}-{:.2f}-{:.3f}/'.format(model_name,
                                                               hps.n_steps_encoder,
                                                               hps.n_steps_decoder,
                                                               hps.n_stacked_layers,
                                                               hps.n_hidden_encoder,
                                                               hps.dropout_rate,
                                                               hps.lambda_l2_reg)
    model_path += 'saved_models/final_model.ckpt'

    # test params
    n_split_test = 50  # times of splitting test set
    # test_rmses = []
    # test_maes = []

    # restore model
    print("Starting loading model...")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model.init(sess)

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model successfully restored from file: %s" % model_path)

        # test
        test_loss = 0
        test_indexes = np.int64(
            np.linspace(0, num_test, n_split_test))
        preds = []
        labels = []
        for k in range(n_split_test - 1):
            feed_dict = my_get_valid_batch_feed_dict(model, test_indexes, k, test_data)
            # re-scale predicted labels
            batch_preds = sess.run(model.phs['preds'], feed_dict)
            batch_preds = np.swapaxes(batch_preds, 0, 1)
            # print(batch_preds.shape)
            batch_preds = np.reshape(batch_preds, [batch_preds.shape[0], -1])
            batch_preds = scaler.inverse_transform(batch_preds)
            # re-scale real labels
            batch_labels = test_data[4]
            batch_labels = batch_labels[test_indexes[k]:test_indexes[k + 1]]
            batch_labels = scaler.inverse_transform(batch_labels)
            # test_rmses.append(root_mean_squared_error(
            #     batch_labels, batch_preds))
            if k == 0:
                preds = batch_preds
                labels = batch_labels
            else:
                preds = np.vstack((preds, batch_preds))
                labels = np.vstack((labels, batch_labels))
            # test_maes.append(Mae(batch_labels, batch_preds))
        np.save('./npy/mae_compare/predict_in_0_hangzhou_GeoMan_day28.npy', preds)
        np.save('./npy/mae_compare/truth_in_0_hangzhou_GeoMan_day28.npy', labels)


def test_TaxiBJ():
    # use specific gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # load hyperparameters
    session = tf.Session(config=tf_config)
    hps = basic_hyperparams()
    hps_dict = json.load(open('./hparam_files/TaxiBJGeoMan.json', 'r'))
    hps.override_from_dict(hps_dict)

    # model construction
    tf.reset_default_graph()
    from GeoMAN import GeoMAN

    print(hps)
    model = GeoMAN(hps)

    # read data from test set
    input_path = './data/'
    val_local_inputs = np.load('./data/test_local_inputs_0_in_day0405.npy')
    val_global_inputs = np.load('./data/test_global_inputs_0_in_day0405.npy')
    val_global_attn_states = np.load('./data/test_global_attn_states_day0405.npy')
    val_external_inputs = np.load('./data/val_external_inputs_taxibj.npy')
    val_labels = np.load('./data/test_labels_0_in_day0405.npy')
    test_data = [val_local_inputs, val_global_inputs, val_global_attn_states, val_external_inputs, val_labels]
    num_test = len(test_data[0])
    print('test samples: {0}'.format(num_test))

    # read scaler of the labels
    f = open('./data/scalers/scaler-0.pkl', 'rb')
    scaler = pickle.load(f)
    f.close()

    # path
    if hps.ext_flag:
        if hps.s_attn_flag == 0:
            model_name = 'GeoMANng'
        elif hps.s_attn_flag == 1:
            model_name = 'GeoMANnl'
        else:
            model_name = 'GeoMAN_TaxiBJ'
    else:
        model_name = 'GeoMANne'
    model_path = './logs/{}-{}-{}-{}-{}-{:.2f}-{:.3f}/'.format(model_name,
                                                               hps.n_steps_encoder,
                                                               hps.n_steps_decoder,
                                                               hps.n_stacked_layers,
                                                               hps.n_hidden_encoder,
                                                               hps.dropout_rate,
                                                               hps.lambda_l2_reg)
    model_path += 'saved_models/final_model.ckpt'

    # test params
    n_split_test = 20  # times of splitting test set
    # test_rmses = []
    # test_maes = []

    # restore model
    print("Starting loading model...")
    saver = tf.train.Saver()
    with tf.Session() as sess:
        model.init(sess)

        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        print("Model successfully restored from file: %s" % model_path)

        # test
        test_loss = 0
        test_indexes = np.int64(
            np.linspace(0, num_test, n_split_test))
        preds = []
        labels = []
        for k in range(n_split_test - 1):
            feed_dict = my_get_valid_batch_feed_dict(model, test_indexes, k, test_data)
            # re-scale predicted labels
            batch_preds = sess.run(model.phs['preds'], feed_dict)
            batch_preds = np.swapaxes(batch_preds, 0, 1)
            # print(batch_preds.shape)
            batch_preds = np.reshape(batch_preds, [batch_preds.shape[0], -1])
            batch_preds = scaler.inverse_transform(batch_preds)
            # re-scale real labels
            batch_labels = test_data[4]
            batch_labels = batch_labels[test_indexes[k]:test_indexes[k + 1]]
            batch_labels = scaler.inverse_transform(batch_labels)
            # test_rmses.append(root_mean_squared_error(
            #     batch_labels, batch_preds))
            if k == 0:
                preds = batch_preds
                labels = batch_labels
            else:
                preds = np.vstack((preds, batch_preds))
                labels = np.vstack((labels, batch_labels))
            # test_maes.append(Mae(batch_labels, batch_preds))
        np.save('./npy/mae_compare/predict_in_0_taxibj_GeoMan_day0405.npy', preds)
        np.save('./npy/mae_compare/truth_in_0_taxibj_GeoMan_day0405.npy', labels)


if __name__ == '__main__':
    test_TaxiBJ()
