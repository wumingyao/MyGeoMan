# -*- coding:utf-8 -*-

import numpy as np

def search_recent_data(train, num_of_hours, label_start_idx, num_for_predict):
    '''
    just like search_day_data, this function search previous hour data
    '''
    if label_start_idx + num_for_predict > len(train):
        return None
    x_idx = []
    for i in range(1, num_of_hours + 1):
        start_idx, end_idx = label_start_idx - i, label_start_idx - i + 1
        if start_idx >= 0 and end_idx >= 0:
            x_idx.append((start_idx, end_idx))
    if len(x_idx) != num_of_hours:
        return None
    return list(reversed(x_idx)), (label_start_idx, label_start_idx + num_for_predict)


def generate_x_y(train, num_of_hours, num_for_predict):
    '''
    Parameters
    ----------
    train: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows)

    num_of_weeks=3, num_of_days=3, num_of_hours=3: int
    # 比如预测周二7：00-7：10的数据
    这里的num_of_weeks=3指的是上周周二{6:40-6:50,6:50-7:00,7:00-7:10},而不是指前三周
    同理num_of_days=3指的是前一天周一{6:40-6:50,6:50-7:00,7:00-7:10},而不是指前三天
    同理num_of_hours=3指的是今天周二{6：30-6：40,6:40-6:50,6:50-7:00}
    num_for_predict=1
    Returns
    ----------
    week_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, num_of_weeks)

    day_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_days)

    recent_data: np.ndarray, shape is (num_of_samples, num_of_stations, num_of_flows, points_per_hour * num_of_hours)

    target: np.ndarray, shape is (num_of_samples, num_of_stations, num_for_predict)

    '''
    length = len(train)
    data = []
    # 根据recent\day\week来重新组织数据集
    for i in range(length):
        recent = search_recent_data(train, num_of_hours, i, num_for_predict)
        if recent:
            recent_data = np.concatenate([train[i: j] for i, j in recent[0]], axis=0)
            data.append((recent_data, train[recent[1][0]: recent[1][1]]))
    features, label = zip(*data)
    return features, label
