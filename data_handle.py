import numpy as np
from libs.utils import generate_x_y
import warnings


# 构建杭州地铁训练验证数据
def handleData(Metro_Flow_Matrix):
    warnings.filterwarnings('ignore')

    N_days = 17  # 用了多少天的数据(目前17个工作日)
    N_hours = 24
    N_time_slice = 6  # 1小时有6个时间片
    N_station = 81  # 81个站点
    N_flow = 2  # 进站 & 出站
    len_seq1 = 2  # week时间序列长度为2
    len_seq2 = 3  # day时间序列长度为3
    len_seq3 = 5  # hour时间序列长度为5
    len_pre = 1
    nb_flow = 2  # 输入特征

    # ——————————————————————————————组织数据———————————————————————————————

    # 生成训练样本（也很关键）
    data, target = generate_x_y(Metro_Flow_Matrix, len_seq3, len_pre)  # type为tuple
    node_data_3 = np.array(data) * 100
    target = np.array(target) * 100
    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    target = np.squeeze(target, axis=1)

    # ——————————————————————————————重新组织数据——————————————————————————————————
    # 将data切割出recent\period\trend数据
    local_inputs = node_data_3[:, :, 0, :]
    np.save('./data/local_inputs_0_in.npy', local_inputs)
    global_inputs = node_data_3[:, :, :, 0]
    np.save('./data/global_inputs_0_in.npy', global_inputs)
    global_attn_states = node_data_3.reshape(
        [node_data_3.shape[0], node_data_3.shape[2], node_data_3.shape[3], node_data_3.shape[1]])
    np.save('./data/global_attn_states.npy', global_attn_states)
    labels = target[:, 0:1, 0:1]
    np.save('./data/labels_0_in.npy', labels)

    external_inputs = np.zeros([node_data_3.shape[0], 1, 5])
    # ——————————————————————————————加入外部信息-周信息——————————————————————————————————

    DAY = 2  # 训练样本从3号开始，为周四，最后需要移动5个时间片[0,1,2,3,4]
    x_external_information1 = np.zeros([N_days * N_hours * N_time_slice, 1])
    # range(start, stop[, step])
    # 0到16
    for i in range(0, N_days * N_hours * N_time_slice, 24 * 6):
        # 标记每个样本属于哪一天
        x_external_information1[i:i + 24 * 6, 0] = DAY
        DAY = (DAY + 1) % 5
    external_inputs[:, :, 0] = x_external_information1[len_seq3:, :]
    # ——————————————————————————————加入外部信息-小时&分钟信息——————————————————————————————————
    HOUR = 0  # 从0时刻开始，最后需要移动5个时间片[0,1,2,...,24*6-1]
    x_external_information2 = np.zeros([N_days * N_hours * N_time_slice, 1])
    for i in range(0, N_days * N_hours * N_time_slice):
        x_external_information2[i, 0] = HOUR
        HOUR = (HOUR + 1) % (24 * 6)
    external_inputs[:, :, 1] = x_external_information2[len_seq3:, :]
    # ——————————————————————————————加入外部信息-天气信息—————————————————————————————————
    x_external_information4 = np.zeros([N_days * N_hours * N_time_slice, 1])
    # [中雨、小雨、阴、多云、晴] --> 简化情况为[雨天/晴天] 0/1
    # 2号--阴
    x_external_information4[0:24 * 6, 0] = 1
    # 3号--小雨/阴
    x_external_information4[24 * 6:2 * 24 * 6, 0] = 0
    # 4号--中雨/小雨
    x_external_information4[2 * 24 * 6:3 * 24 * 6, 0] = 0
    # 7号--小雨
    x_external_information4[3 * 24 * 6:4 * 24 * 6, 0] = 0
    # 8号--小雨/阴
    x_external_information4[4 * 24 * 6:5 * 24 * 6, 0] = 0
    # 9号--中雨/小雨
    x_external_information4[5 * 24 * 6:6 * 24 * 6, 0] = 0
    # 10号--小雨
    x_external_information4[6 * 24 * 6:7 * 24 * 6, 0] = 0
    # 11号--小雨
    x_external_information4[7 * 24 * 6:8 * 24 * 6, 0] = 0
    # 14号--小雨
    x_external_information4[8 * 24 * 6:9 * 24 * 6, 0] = 0
    # 15号--小雨
    x_external_information4[9 * 24 * 6:10 * 24 * 6, 0] = 0
    # 16号--多云
    x_external_information4[10 * 24 * 6:11 * 24 * 6, 0] = 1
    # 17号--晴
    x_external_information4[11 * 24 * 6:12 * 24 * 6, 0] = 1
    # 18号--小雨
    x_external_information4[12 * 24 * 6:13 * 24 * 6, 0] = 0
    # 21号--多云/晴
    x_external_information4[13 * 24 * 6:14 * 24 * 6, 0] = 1
    # 22号--晴
    x_external_information4[14 * 24 * 6:15 * 24 * 6, 0] = 1
    # 23号--晴
    x_external_information4[15 * 24 * 6:16 * 24 * 6, 0] = 1
    # 24号--晴
    x_external_information4[16 * 24 * 6:17 * 24 * 6, 0] = 1
    # 25号--多云
    # x_external_information4[17*24*6:18*24*6, 3] = 1
    # 28号
    # x_external_information4[18*24*6:19*24*6, 2] = 1
    # 除去2号的天气信息，此处应该是开始矩阵大小没设计好，所以需要往后移动144个时间片，然后再移动5个时间片
    # x_external_information4 = x_external_information4[24 * 6:, :]
    external_inputs[:, :, 2] = x_external_information4[len_seq3:, :]
    # ——————————————————————————————加入外部信息-闸机数量—————————————————————————————————
    external_information5 = np.load('./npy/train_data/Two_more_features.npy')
    x_external_information5 = np.zeros([N_days * N_hours * N_time_slice, 81])
    external_information5 = external_information5[:, 0]
    for i in range(N_days * N_hours * N_time_slice):
        x_external_information5[i, :] = external_information5
    external_inputs[:, :, 3] = x_external_information5[len_seq3:, 0:1]
    # ——————————————————————————————加入早晚高峰、一般高峰、平峰信息————————————————————————————————
    x_external_information9 = np.zeros([N_days * N_hours * N_time_slice, 1])
    # [平峰、一般高峰、早晚高峰] [0,1,2]
    for i in range(0, N_days * N_hours * N_time_slice, 24 * 6):
        # ——————————————————早晚高峰—————————————————————
        x_external_information9[i + 39:i + 54, 0] = 2  # 7：30 - 9：00
        x_external_information9[i + 102:i + 114, 0] = 2  # 17：00 - 19：00
        # ——————————————————高峰—————————————————————————
        x_external_information9[i + 33:i + 39, 0] = 1  # 6:30-7:30
        x_external_information9[i + 63:i + 70, 0] = 1  # 10:30-11:30
        x_external_information9[i + 99:i + 102, 0] = 1  # 16:30-17:30
        x_external_information9[i + 114:i + 132, 0] = 1  # 19:00-22:00
    external_inputs[:, :, 4] = x_external_information9[len_seq3:, :]
    np.save('./data/external_inputs.npy', external_inputs)

    # ————————————————————————————————构建验证集合(24-25号数据作为验证集)—————————————————————————————————————
    # 构造得有点复杂.....2019.05.22
    node_day_25 = np.load('./npy/train_data/day_25.npy')
    val_node_data = node_day_25

    # normalization

    # 构建好验证集的样本和标签
    val_node_data, val_node_target = generate_x_y(val_node_data, len_seq3, len_pre)

    val_node_data_3 = np.array(val_node_data)

    val_node_target = np.array(val_node_target)
    val_node_target = np.squeeze(val_node_target, axis=1)

    val_local_inputs = val_node_data_3[:, :, 0, :]
    np.save('./data/val_local_inputs_0_in.npy', val_local_inputs)
    val_global_inputs = val_node_data_3[:, :, :, 0]
    np.save('./data/val_global_inputs_0_in.npy', val_global_inputs)
    val_global_attn_states = val_node_data_3.reshape(
        [val_node_data_3.shape[0], val_node_data_3.shape[2], val_node_data_3.shape[3], val_node_data_3.shape[1]])
    np.save('./data/val_global_attn_states.npy', val_global_attn_states)
    val_labels = val_node_target[:, 0:1, 0:1]
    np.save('./data/val_labels_0_in.npy', val_labels)

    val_external_inputs = np.zeros([val_node_data_3.shape[0], 1, 5])

    # ——————————————————————添加验证集外部信息————————————————————————
    # 默认了00:00-00:40的人流量均为0
    # 0125是周五,weekday信息
    x_val_external_information1 = np.zeros([24 * 6, 1])
    x_val_external_information1[:, 0] = 4  # 代表周五
    val_external_inputs[:, :, 0] = x_val_external_information1[len_seq3:, :]
    # 时间片信息
    x_val_external_information2 = np.zeros([24 * 6, 1])
    HOUR = 0
    for i in range(0, 24 * 6):
        x_val_external_information2[i, 0] = HOUR
        HOUR = HOUR + 1
    val_external_inputs[:, :, 1] = x_val_external_information2[len_seq3:, :]
    # 天气信息,25号为多云
    x_val_external_information4 = np.zeros([24 * 6, 1])
    x_val_external_information4[:, 0] = 1
    val_external_inputs[:, :, 2] = x_val_external_information4[len_seq3:, :]
    # 闸机信息
    x_val_external_information5 = np.zeros([24 * 6, 81])
    t = np.load('./npy/train_data/Two_more_features.npy')
    t = t[:, 0]
    for i in range(144):
        x_val_external_information5[i, :] = t
    val_external_inputs[:, :, 3] = x_val_external_information5[len_seq3:, 0:1]
    # 早晚高峰、高峰、平峰信息
    x_val_external_information9 = np.zeros([N_hours * N_time_slice, 1])
    # #——————————————————早晚高峰—————————————————————
    x_val_external_information9[39:54, 0] = 2  # 7：30 - 9：00
    x_val_external_information9[102:114, 0] = 2  # 17：00 - 19：00
    # #——————————————————高峰—————————————————————————
    x_val_external_information9[33:39, 0] = 1  # 6:30-7:30
    x_val_external_information9[63:70, 0] = 1  # 10:30-11:30
    x_val_external_information9[99:102, 0] = 1  # 16:30-17:30
    x_val_external_information9[114:132, 0] = 1  # 19:00-22:00

    val_external_inputs[:, :, 4] = x_val_external_information9[len_seq3:, :]
    np.save('./data/val_external_inputs.npy', val_external_inputs)


# 构建测试集
def test_dataSet_handle(node_day, day):
    # 构建测试集
    # node_day = np.load('./npy/test_data/raw_node_data_day19.npy')

    # 构建好验证集的样本和标签
    val_node_data, val_node_target = generate_x_y(node_day, 5, 1)

    val_node_data_3 = np.array(val_node_data)

    val_node_target = np.array(val_node_target)
    val_node_target = np.squeeze(val_node_target, axis=1)

    val_local_inputs = val_node_data_3[:, :, 0, :]
    np.save('./data/test_local_inputs_0_in_day' + day + '.npy', val_local_inputs)
    val_global_inputs = val_node_data_3[:, :, :, 0]
    np.save('./data/test_global_inputs_0_in_day' + day + '.npy', val_global_inputs)
    val_global_attn_states = val_node_data_3.reshape(
        [val_node_data_3.shape[0], val_node_data_3.shape[2], val_node_data_3.shape[3], val_node_data_3.shape[1]])
    np.save('./data/test_global_attn_states_day' + day + '.npy', val_global_attn_states)
    val_labels = val_node_target[:, 0:1, 0:1]
    np.save('./data/test_labels_0_in_day' + day + '.npy', val_labels)


# 构建TaxiBJ训练验证数据
def handleData_TaxiBJ(Metro_Flow_Matrix):
    warnings.filterwarnings('ignore')

    N_days = 31  # 用了多少天的数据(目前17个工作日)
    N_hours = 24
    N_time_slice = 2  # 1小时有6个时间片
    len_seq3 = 5  # hour时间序列长度为5
    len_pre = 1

    # ——————————————————————————————组织数据———————————————————————————————

    # 生成训练样本（也很关键）
    data, target = generate_x_y(Metro_Flow_Matrix, len_seq3, len_pre)  # type为tuple
    node_data_3 = np.array(data) * 100
    target = np.array(target) * 100
    # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉
    target = np.squeeze(target, axis=1)

    # ——————————————————————————————重新组织数据——————————————————————————————————
    # 将data切割出recent\period\trend数据
    local_inputs = node_data_3[:, :, 0, :]
    np.save('./data/local_inputs_0_in_taxibj.npy', local_inputs)
    global_inputs = node_data_3[:, :, :, 0]
    print(global_inputs.shape)
    np.save('./data/global_inputs_0_in_taxibj.npy', global_inputs)
    global_attn_states = node_data_3.reshape(
        [node_data_3.shape[0], node_data_3.shape[2], node_data_3.shape[3], node_data_3.shape[1]])
    np.save('./data/global_attn_states_taxibj.npy', global_attn_states)
    labels = target[:, 0:1, 0:1]
    np.save('./data/labels_0_in_taxibj.npy', labels)

    external_inputs = np.zeros([node_data_3.shape[0], 1, 3])
    # ——————————————————————————————加入外部信息-周信息——————————————————————————————————

    DAY = 2
    x_external_information1 = np.zeros([N_days * N_hours * N_time_slice, 1])
    # range(start, stop[, step])
    # 0到16
    for i in range(0, N_days * N_hours * N_time_slice, N_hours * N_time_slice):
        # 标记每个样本属于哪一天
        x_external_information1[i:i + N_hours * N_time_slice, 0] = DAY
        DAY = (DAY + 1) % 5
    external_inputs[:, :, 0] = x_external_information1[len_seq3:, :]
    # ——————————————————————————————加入外部信息-小时&分钟信息——————————————————————————————————
    HOUR = 0  # 从0时刻开始，最后需要移动5个时间片[0,1,2,...,24*6-1]
    x_external_information2 = np.zeros([N_days * N_hours * N_time_slice, 1])
    for i in range(0, N_days * N_hours * N_time_slice):
        x_external_information2[i, 0] = HOUR
        HOUR = (HOUR + 1) % (N_hours * N_time_slice)
    external_inputs[:, :, 1] = x_external_information2[len_seq3:, :]
    # ——————————————————————————————加入早晚高峰、一般高峰、平峰信息————————————————————————————————
    x_external_information9 = np.zeros([N_days * N_hours * N_time_slice, 1])
    # [平峰、一般高峰、早晚高峰] [0,1,2]
    for i in range(0, N_days * N_hours * N_time_slice, N_hours * N_time_slice):
        # ——————————————————早晚高峰—————————————————————
        x_external_information9[i + 39:i + 54, 0] = 2  # 7：30 - 9：00
        x_external_information9[i + 102:i + 114, 0] = 2  # 17：00 - 19：00
        # ——————————————————高峰—————————————————————————
        x_external_information9[i + 33:i + 39, 0] = 1  # 6:30-7:30
        x_external_information9[i + 63:i + 70, 0] = 1  # 10:30-11:30
        x_external_information9[i + 99:i + 102, 0] = 1  # 16:30-17:30
        x_external_information9[i + 114:i + 132, 0] = 1  # 19:00-22:00
    external_inputs[:, :, 2] = x_external_information9[len_seq3:, :]
    np.save('./data/external_inputs_TaxiBJ.npy', external_inputs)

    # ————————————————————————————————构建验证集合(24-25号数据作为验证集)—————————————————————————————————————
    # 构造得有点复杂.....2019.05.22
    node_day_25 = np.load('./npy/test_data/taxibj_node_data_day0404.npy')[:, 0:81, :]
    val_node_data = node_day_25

    # normalization

    # 构建好验证集的样本和标签
    val_node_data, val_node_target = generate_x_y(val_node_data, len_seq3, len_pre)

    val_node_data_3 = np.array(val_node_data)

    val_node_target = np.array(val_node_target)
    val_node_target = np.squeeze(val_node_target, axis=1)

    val_local_inputs = val_node_data_3[:, :, 0, :]
    np.save('./data/val_local_inputs_0_in_taxibj_0404.npy', val_local_inputs)
    val_global_inputs = val_node_data_3[:, :, :, 0]
    np.save('./data/val_global_inputs_0_in_taxibj_0404.npy', val_global_inputs)
    val_global_attn_states = val_node_data_3.reshape(
        [val_node_data_3.shape[0], val_node_data_3.shape[2], val_node_data_3.shape[3], val_node_data_3.shape[1]])
    np.save('./data/val_global_attn_states_taxibj_0404.npy', val_global_attn_states)
    val_labels = val_node_target[:, 0:1, 0:1]
    np.save('./data/val_labels_0_in_taxibj_0404.npy', val_labels)

    val_external_inputs = np.zeros([val_node_data_3.shape[0], 1, 3])

    # ——————————————————————添加验证集外部信息————————————————————————
    # 默认了00:00-00:40的人流量均为0
    # 0125是周五,weekday信息
    x_val_external_information1 = np.zeros([N_hours * N_time_slice, 1])
    x_val_external_information1[:, 0] = 1
    val_external_inputs[:, :, 0] = x_val_external_information1[len_seq3:, :]
    # 时间片信息
    x_val_external_information2 = np.zeros([N_hours * N_time_slice, 1])
    HOUR = 0
    for i in range(0, N_hours * N_time_slice):
        x_val_external_information2[i, 0] = HOUR
        HOUR = HOUR + 1
    val_external_inputs[:, :, 1] = x_val_external_information2[len_seq3:, :]

    # 早晚高峰、高峰、平峰信息
    x_val_external_information9 = np.zeros([N_hours * N_time_slice, 1])
    # #——————————————————早晚高峰—————————————————————
    x_val_external_information9[39:54, 0] = 2  # 7：30 - 9：00
    x_val_external_information9[102:114, 0] = 2  # 17：00 - 19：00
    # #——————————————————高峰—————————————————————————
    x_val_external_information9[33:39, 0] = 1  # 6:30-7:30
    x_val_external_information9[63:70, 0] = 1  # 10:30-11:30
    x_val_external_information9[99:102, 0] = 1  # 16:30-17:30
    x_val_external_information9[114:132, 0] = 1  # 19:00-22:00

    val_external_inputs[:, :, 2] = x_val_external_information9[len_seq3:, :]
    np.save('./data/val_external_inputs_taxibj.npy', val_external_inputs)


if __name__ == '__main__':
    # handleData(np.load('./npy/train_data/raw_node_data.npy'))
    # test_dataSet_handle(np.load('./npy/test_data/raw_node_data_day19.npy'), '19')
    # test_dataSet_handle(np.load('./npy/test_data/raw_node_data_day20.npy'), '20')
    # test_dataSet_handle(np.load('./npy/test_data/raw_node_data_day25.npy'), '25')
    # test_dataSet_handle(np.load('./npy/test_data/raw_node_data_day28.npy'), '28')
    test_dataSet_handle(np.load('./npy/test_data/taxibj_node_data_day0402.npy')[:, 0:81:], '0402')
    test_dataSet_handle(np.load('./npy/test_data/taxibj_node_data_day0403.npy')[:, 0:81:], '0403')
    test_dataSet_handle(np.load('./npy/test_data/taxibj_node_data_day0404.npy')[:, 0:81:], '0404')
    test_dataSet_handle(np.load('./npy/test_data/taxibj_node_data_day0405.npy')[:, 0:81:], '0405')
    # node_data = np.load('./npy/train_data/taxibj_node_data_month3_23.npy')
    # node_data = node_data.reshape([node_data.shape[0], node_data.shape[2]*node_data.shape[3], node_data.shape[1]])[:, 0:81, :]
    # handleData_TaxiBJ(node_data)
