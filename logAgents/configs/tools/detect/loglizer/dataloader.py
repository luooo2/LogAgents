"""
The interface to load log datasets. The datasets currently supported include
HDFS and BGL.

Authors:
    LogPAI Team

"""

import pandas as pd
import os
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import OrderedDict

def _split_data(x_data, y_data=None, train_ratio=0, split_type='uniform'):
    if split_type == 'uniform' and y_data is not None:
        pos_idx = y_data > 0
        x_pos = x_data[pos_idx]
        y_pos = y_data[pos_idx]
        x_neg = x_data[~pos_idx]
        y_neg = y_data[~pos_idx]
        train_pos = int(train_ratio * x_pos.shape[0])
        train_neg = int(train_ratio * x_neg.shape[0])
        x_train = np.hstack([x_pos[0:train_pos], x_neg[0:train_neg]])
        y_train = np.hstack([y_pos[0:train_pos], y_neg[0:train_neg]])
        x_test = np.hstack([x_pos[train_pos:], x_neg[train_neg:]])
        y_test = np.hstack([y_pos[train_pos:], y_neg[train_neg:]])
    elif split_type == 'sequential':
        num_train = int(train_ratio * x_data.shape[0])
        x_train = x_data[0:num_train]
        x_test = x_data[num_train:]
        if y_data is None:
            y_train = None
            y_test = None
        else:
            y_train = y_data[0:num_train]
            y_test = y_data[num_train:]
    # Random shuffle
    indexes = shuffle(np.arange(x_train.shape[0]))
    x_train = x_train[indexes]
    if y_train is not None:
        y_train = y_train[indexes]
    return (x_train, y_train), (x_test, y_test)

def load_HDFS(log_file, label_file=None, window='session', train_ratio=0.5, split_type='sequential', save_csv=False, window_size=0):
    """ Load HDFS structured log into train and test data

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
    """

    print('====== Input data summary ======')

    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)

    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        print("Loading", log_file)
        struct_log = pd.read_csv(log_file, engine='c',
                na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)
            for blk_Id in blkId_set:
                if not blk_Id in data_dict:
                    data_dict[blk_Id] = []
                data_dict[blk_Id].append(row['EventId'])
        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
        
        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

            # Split train and test data
            (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values, 
                data_df['Label'].values, train_ratio, split_type)
        
            print(y_train.sum(), y_test.sum())

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        if window_size > 0:
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            print(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1-y_train).sum(), y_train.shape[0]))
            print(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1-y_test).sum(), y_test.shape[0]))
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                  x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train), (x_test, y_test)

def slice_hdfs(x, y, window_size):
    results_data = []
    print("Slicing {} sessions, with window {}".format(x.shape[0], window_size))
    for idx, sequence in enumerate(x):
        seqlen = len(sequence)
        i = 0
        while (i + window_size) < seqlen:
            slice = sequence[i: i + window_size]
            results_data.append([idx, slice, sequence[i + window_size], y[idx]])
            i += 1
        else:
            slice = sequence[i: i + window_size]
            slice += ["#Pad"] * (window_size - len(slice))
            results_data.append([idx, slice, "#Pad", y[idx]])
    results_df = pd.DataFrame(results_data, columns=["SessionId", "EventSequence", "Label", "SessionLabel"])
    print("Slicing done, {} windows generated".format(results_df.shape[0]))
    return results_df[["SessionId", "EventSequence"]], results_df["Label"], results_df["SessionLabel"]



def load_BGL(log_file, label_file=None, window='sliding', time_interval=60, stepping_size=60, 
             train_ratio=0.8):
    """  TODO

    """


def bgl_preprocess_data(para, raw_data, event_mapping_data):
    """ split logs into sliding windows, built an event count matrix and get the corresponding label

    Args:
    --------
    para: the parameters dictionary
    raw_data: list of (label, time)
    event_mapping_data: a list of event index, where each row index indicates a corresponding log

    Returns:
    --------
    event_count_matrix: event count matrix, where each row is an instance (log sequence vector)
    labels: a list of labels, 1 represents anomaly
    """

    # create the directory for saving the sliding windows (start_index, end_index), which can be directly loaded in future running
    if not os.path.exists(para['save_path']):
        os.mkdir(para['save_path'])
    log_size = raw_data.shape[0]
    sliding_file_path = para['save_path']+'sliding_'+str(para['window_size'])+'h_'+str(para['step_size'])+'h.csv'

    #=============divide into sliding windows=========#
    start_end_index_list = [] # list of tuples, tuple contains two number, which represent the start and end of sliding time window
    label_data, time_data = raw_data[:,0], raw_data[:, 1]
    if not os.path.exists(sliding_file_path):
        # split into sliding window
        start_time = time_data[0]
        start_index = 0
        end_index = 0

        # get the first start, end index, end time
        for cur_time in time_data:
            if  cur_time < start_time + para['window_size']*3600:
                end_index += 1
                end_time = cur_time
            else:
                start_end_pair=tuple((start_index,end_index))
                start_end_index_list.append(start_end_pair)
                break
        # move the start and end index until next sliding window
        while end_index < log_size:
            start_time = start_time + para['step_size']*3600
            end_time = end_time + para['step_size']*3600
            for i in range(start_index,end_index):
                if time_data[i] < start_time:
                    i+=1
                else:
                    break
            for j in range(end_index, log_size):
                if time_data[j] < end_time:
                    j+=1
                else:
                    break
            start_index = i
            end_index = j
            start_end_pair = tuple((start_index, end_index))
            start_end_index_list.append(start_end_pair)
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset\n'%inst_number)
        np.savetxt(sliding_file_path,start_end_index_list,delimiter=',',fmt='%d')
    else:
        print('Loading start_end_index_list from file')
        start_end_index_list = pd.read_csv(sliding_file_path, header=None).values
        inst_number = len(start_end_index_list)
        print('there are %d instances (sliding windows) in this dataset' % inst_number)

    # get all the log indexes in each time window by ranging from start_index to end_index
    expanded_indexes_list=[]
    for t in range(inst_number):
        index_list = []
        expanded_indexes_list.append(index_list)
    for i in range(inst_number):
        start_index = start_end_index_list[i][0]
        end_index = start_end_index_list[i][1]
        for l in range(start_index, end_index):
            expanded_indexes_list[i].append(l)

    event_mapping_data = [row[0] for row in event_mapping_data]
    event_num = len(list(set(event_mapping_data)))
    print('There are %d log events'%event_num)

    #=============get labels and event count of each sliding window =========#
    labels = []
    event_count_matrix = np.zeros((inst_number,event_num))
    for j in range(inst_number):
        label = 0   #0 represent success, 1 represent failure
        for k in expanded_indexes_list[j]:
            event_index = event_mapping_data[k]
            event_count_matrix[j, event_index] += 1
            if label_data[k]:
                label = 1
                continue
        labels.append(label)
    assert inst_number == len(labels)
    print("Among all instances, %d are anomalies"%sum(labels))
    assert event_count_matrix.shape[0] == len(labels)
    return event_count_matrix, labels

# def load_HDFS_with_raw(log_file, label_file=None, window='session', train_ratio=0.5,
#                        split_type='sequential', save_csv=False, window_size=0):
#     """
#     Load HDFS structured log into train and test data with raw log details
#
#     Returns:
#         (x_train, y_train): training event sequences and labels
#         (x_test, y_test): testing event sequences and labels
#         raw_logs: dict with keys 'train' and 'test' containing raw log details
#     """
#     print('====== Loading data with raw log details ======')
#
#     # 存储原始日志数据的结构
#     raw_logs = {'train': [], 'test': []}
#
#     # 1. 加载结构化日志并保留原始信息
#     print("Loading structured log:", log_file)
#     struct_log = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
#
#     # 2. 创建更详细的数据字典
#     data_dict = OrderedDict()
#     raw_log_dict = OrderedDict()  # 存储每个BlockId的原始日志
#
#     for idx, row in struct_log.iterrows():
#         blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
#         blkId_set = set(blkId_list)
#
#         for blk_Id in blkId_set:
#             # 保存事件序列
#             if blk_Id not in data_dict:
#                 data_dict[blk_Id] = []
#                 raw_log_dict[blk_Id] = []
#
#             data_dict[blk_Id].append(row['EventId'])
#
#             # 保存原始日志详细信息
#             raw_log_dict[blk_Id].append({
#                 'EventId': row['EventId'],
#                 'Content': row['Content'],
#                 'Time': row.get('Time', ''),
#                 'Date': row.get('Date', ''),
#                 'Pid':row.get('Pid',''),
#                 'Level':row.get('Level',''),
#                 'Component': row.get('Component', ''),
#                 'EventTemplate':row.get('EventTemplate','')
#             })
#
#     # 3. 创建DataFrame
#     data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
#     data_df['RawLogs'] = data_df['BlockId'].map(lambda x: raw_log_dict[x])
#
#     # 4. 加载标签（如果有）
#     label_data = None
#     if label_file:
#         print("Loading labels:", label_file)
#         label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
#         label_data = label_data.set_index('BlockId')
#         label_dict = label_data['Label'].to_dict()
#         data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict.get(x, '') == 'Anomaly' else 0)
#
#     # 5. 分割数据集
#     if label_file:
#         # 获取索引用于分割
#         indices = np.arange(len(data_df))
#
#         if split_type == 'uniform':
#             # 均匀分割正负样本
#             pos_idx = data_df.index[data_df['Label'] == 1]
#             neg_idx = data_df.index[data_df['Label'] == 0]
#
#             # 分割正样本
#             pos_train, pos_test = train_test_split(
#                 pos_idx, train_size=train_ratio, random_state=42
#             )
#             # 分割负样本
#             neg_train, neg_test = train_test_split(
#                 neg_idx, train_size=train_ratio, random_state=42
#             )
#
#             train_idx = np.concatenate([pos_train, neg_train])
#             test_idx = np.concatenate([pos_test, neg_test])
#
#         else:  # sequential
#             # 顺序分割
#             n_train = int(len(data_df) * train_ratio)
#             train_idx = indices[:n_train]
#             test_idx = indices[n_train:]
#
#         # 创建训练集
#         x_train = data_df.loc[train_idx, 'EventSequence'].values
#         y_train = data_df.loc[train_idx, 'Label'].values
#         raw_logs['train'] = data_df.loc[train_idx, 'RawLogs'].tolist()
#
#         # 创建测试集
#         x_test = data_df.loc[test_idx, 'EventSequence'].values
#         y_test = data_df.loc[test_idx, 'Label'].values
#         raw_logs['test'] = data_df.loc[test_idx, 'RawLogs'].tolist()
#
#         print(f"Train anomalies: {y_train.sum()}/{len(y_train)}")
#         print(f"Test anomalies: {y_test.sum()}/{len(y_test)}")
#
#         return (x_train, y_train), (x_test, y_test), raw_logs
#
#     else:
#         # 无标签数据的情况
#         x_data = data_df['EventSequence'].values
#         n_train = int(len(x_data) * train_ratio)
#
#         x_train = x_data[:n_train]
#         x_test = x_data[n_train:]
#
#         raw_logs['train'] = data_df['RawLogs'].iloc[:n_train].tolist()
#         raw_logs['test'] = data_df['RawLogs'].iloc[n_train:].tolist()
#
#         print(f"Total sessions: {len(x_data)}")
#         print(f"Train sessions: {len(x_train)}")
#         print(f"Test sessions: {len(x_test)}")
#
#     return (x_train, None), (x_test, None), raw_logs

def _get_split_indices(n_samples, train_ratio=0.5, split_type='sequential', labels=None):
    """
    获取训练集和测试集的索引

    参数:
    n_samples: 总样本数
    train_ratio: 训练集比例 (0-1)
    split_type: 分割类型 ('uniform' 或 'sequential')
    labels: 标签数组 (用于 'uniform' 分割)

    返回:
    train_indices: 训练样本索引
    test_indices: 测试样本索引
    """
    indices = np.arange(n_samples)

    if split_type == 'uniform' and labels is not None:
        # 分层抽样：保持正负样本比例
        pos_indices = np.where(labels > 0)[0]
        neg_indices = np.where(labels == 0)[0]

        # 计算每类样本的训练数量
        train_pos = int(train_ratio * len(pos_indices))
        train_neg = int(train_ratio * len(neg_indices))

        # 分割正样本
        train_pos_indices = pos_indices[:train_pos]
        test_pos_indices = pos_indices[train_pos:]

        # 分割负样本
        train_neg_indices = neg_indices[:train_neg]
        test_neg_indices = neg_indices[train_neg:]

        # 合并索引
        train_indices = np.concatenate([train_pos_indices, train_neg_indices])
        test_indices = np.concatenate([test_pos_indices, test_neg_indices])

    elif split_type == 'sequential':
        # 顺序分割
        num_train = int(train_ratio * n_samples)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]

    else:
        raise ValueError(f"Unsupported split_type: {split_type} or missing labels for uniform split")

    # 随机打乱训练集索引（保持与 _split_data 一致）
    train_indices = shuffle(train_indices)

    return train_indices, test_indices


def load_HDFS_with_raw(log_file, label_file=None, window='session', train_ratio=0.5, split_type='sequential', save_csv=False,
              window_size=0):
    """ Load HDFS structured log into train and test data and raw logs

    Arguments
    ---------
        log_file: str, the file path of structured log.
        label_file: str, the file path of anomaly labels, None for unlabeled data
        window: str, the window options including `session` (default).
        train_ratio: float, the ratio of training data for train/test split.
        split_type: `uniform` or `sequential`, which determines how to split dataset. `uniform` means
            to split positive samples and negative samples equally when setting label_file. `sequential`
            means to split the data sequentially without label_file. That is, the first part is for training,
            while the second part is for testing.

    Returns
    -------
        (x_train, y_train): the training data
        (x_test, y_test): the testing data
        raw_logs: the complete raw data before splitting
    """

    print('====== Input data summary ======')
    raw_data = None

    if log_file.endswith('.npz'):
        # Split training and validation set in a class-uniform way
        data = np.load(log_file)
        x_data = data['x_data']
        y_data = data['y_data']
        (x_train, y_train), (x_test, y_test) = _split_data(x_data, y_data, train_ratio, split_type)
        raw_data = data

    # 暂且只考虑csv格式数据
    elif log_file.endswith('.csv'):
        assert window == 'session', "Only window=session is supported for HDFS dataset."
        print("Loading", log_file)
        struct_log = pd.read_csv(log_file, engine='c',
                                 na_filter=False, memory_map=True)
        data_dict = OrderedDict()
        raw_log_dict = OrderedDict()

        for idx, row in struct_log.iterrows():
            blkId_list = re.findall(r'(blk_-?\d+)', row['Content'])
            blkId_set = set(blkId_list)

            for blk_Id in blkId_set:
                # 保存事件序列
                if blk_Id not in data_dict:
                    data_dict[blk_Id] = []
                    raw_log_dict[blk_Id] = []

                data_dict[blk_Id].append(row['EventId'])

                # 保存原始日志详细信息
                raw_log_dict[blk_Id].append({
                    'EventId': row['EventId'],
                    'Content': row['Content'],
                    'Time': row.get('Time', ''),
                    'Date': row.get('Date', ''),
                    'Pid':row.get('Pid',''),
                    'Level':row.get('Level',''),
                    'Component': row.get('Component', ''),
                    'EventTemplate':row.get('EventTemplate','')
                })

        data_df = pd.DataFrame(list(data_dict.items()), columns=['BlockId', 'EventSequence'])
        # 创建原始日志信息DataFrame
        raw_log_info_list = []
        for block_id, logs in raw_log_dict.items():
            for log in logs:
                log['BlockId'] = block_id
                raw_log_info_list.append(log)

        raw_log_info_df = pd.DataFrame(raw_log_info_list)
        raw_data = raw_log_info_df.copy()

        if label_file:
            # Split training and validation set in a class-uniform way
            label_data = pd.read_csv(label_file, engine='c', na_filter=False, memory_map=True)
            label_data = label_data.set_index('BlockId')
            label_dict = label_data['Label'].to_dict()
            data_df['Label'] = data_df['BlockId'].apply(lambda x: 1 if label_dict[x] == 'Anomaly' else 0)

            # Split train and test data
            # 获取数据数组
            event_sequences = data_df['EventSequence'].values
            labels = data_df['Label'].values
            block_ids = data_df['BlockId'].values

            # 使用 _get_split_indices 获取索引
            train_indices, test_indices = _get_split_indices(
                n_samples=len(event_sequences),
                train_ratio=train_ratio,
                split_type=split_type,
                labels=labels
            )

            # 使用索引分割数据
            x_train = event_sequences[train_indices]
            y_train = labels[train_indices]
            train_block_ids = block_ids[train_indices]

            x_test = event_sequences[test_indices]
            y_test = labels[test_indices]
            test_block_ids = block_ids[test_indices]
            # (x_train, y_train), (x_test, y_test) = _split_data(data_df['EventSequence'].values,
            #                                                    data_df['Label'].values, train_ratio, split_type)

            print(y_train.sum(), y_test.sum())

        if save_csv:
            data_df.to_csv('data_instances.csv', index=False)

        if window_size > 0:
            x_train, window_y_train, y_train = slice_hdfs(x_train, y_train, window_size)
            x_test, window_y_test, y_test = slice_hdfs(x_test, y_test, window_size)
            log = "{} {} windows ({}/{} anomaly), {}/{} normal"
            print(log.format("Train:", x_train.shape[0], y_train.sum(), y_train.shape[0], (1 - y_train).sum(),
                             y_train.shape[0]))
            print(log.format("Test:", x_test.shape[0], y_test.sum(), y_test.shape[0], (1 - y_test).sum(),
                             y_test.shape[0]))
            return (x_train, window_y_train, y_train), (x_test, window_y_test, y_test)

        if label_file is None:
            if split_type == 'uniform':
                split_type = 'sequential'
                print('Warning: Only split_type=sequential is supported \
                if label_file=None.'.format(split_type))
            # Split training and validation set sequentially
            x_data = data_df['EventSequence'].values
            (x_train, _), (x_test, _) = _split_data(x_data, train_ratio=train_ratio, split_type=split_type)
            print('Total: {} instances, train: {} instances, test: {} instances'.format(
                x_data.shape[0], x_train.shape[0], x_test.shape[0]))
            return (x_train, None), (x_test, None), data_df
    else:
        raise NotImplementedError('load_HDFS() only support csv and npz files!')

    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    num_total = num_train + num_test
    num_train_pos = sum(y_train)
    num_test_pos = sum(y_test)
    num_pos = num_train_pos + num_test_pos

    print('Total: {} instances, {} anomaly, {} normal' \
          .format(num_total, num_pos, num_total - num_pos))
    print('Train: {} instances, {} anomaly, {} normal' \
          .format(num_train, num_train_pos, num_train - num_train_pos))
    print('Test: {} instances, {} anomaly, {} normal\n' \
          .format(num_test, num_test_pos, num_test - num_test_pos))

    return (x_train, y_train, train_block_ids), (x_test, y_test, test_block_ids), raw_data
