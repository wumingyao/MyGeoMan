import unittest

import numpy as  np


def Mae(truth, predict):
    # 后处理计算
    truth = truth.flatten()
    predict = predict.flatten()
    loss_matrix = np.abs(truth - predict)
    mae = loss_matrix.sum() / truth.size
    return mae


def Mape(truth, predict):
    # 后处理计算
    truth = truth.flatten()
    predict = predict.flatten()
    loss_matrix = np.abs(truth - predict)
    loss_matrix2 = loss_matrix / truth
    mape = loss_matrix2.sum() / truth.size
    return mape


def Made(truth, predict):
    # 后处理计算
    truth = truth.flatten()
    predict = predict.flatten()
    made = np.median(np.abs(truth - predict))
    return made
