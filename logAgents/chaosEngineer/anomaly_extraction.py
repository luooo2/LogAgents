# import numpy as np
# import pandas as pd
#
# from logAgents.configs.tools.detect.loglizer.models import SVM
# # from logAgents.configs.tools.detect.loglizer.models import LogClustering
# from logAgents.configs.tools.detect.loglizer import dataloader, preprocessing
#
# # def anomaly_detect():
# #     # print("log anomaly detecting……\n")
# #
# #     structured_log = r'E:\swarm\swarm\logAgents\data\logs\HDFS_100k.log_structured.csv' # The structured log file
# #     label_file = r'E:\swarm\swarm\logAgents\data\logs\anomaly_label_100k.csv' # The anomaly label file
# #     max_dist = 0.3  # the threshold to stop the clustering process
# #     anomaly_threshold = 0.3  # the threshold for anomaly detect
# #     (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(structured_log,
# #                                                                 label_file=label_file,
# #                                                                 window='session',
# #                                                                 train_ratio=0.5,
# #                                                                 split_type='uniform')
# #
# #     feature_extractor = preprocessing.FeatureExtractor()
# #     x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
# #     x_test = feature_extractor.transform(x_test)
# #
# #     # SVM 模型
# #     model = SVM()
# #     model.fit(x_train, y_train)
# #
# #     # LogClustering 模型
# #     # model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
# #     # model.fit(x_train[y_train == 0, :])  # Use only normal samples for training
# #
# #
# #     print('Train validation:')
# #     precision, recall, f1 = model.evaluate(x_train, y_train)
# #
# #     print('Test validation:')
# #     precision, recall, f1 = model.evaluate(x_test, y_test)
#
#
# def anomaly_detect_with_analysis():
#     structured_log = r'E:\swarm\swarm\logAgents\data\logs\HDFS_100k.log_structured.csv'
#     label_file = r'E:\swarm\swarm\logAgents\data\logs\anomaly_label_100k.csv'
#
#     # 1. 加载数据并保留原始日志信息
#     (x_train, y_train), (x_test, y_test), raw_logs = dataloader.load_HDFS_with_raw(
#         structured_log,
#         label_file=label_file,
#         window='session',
#         train_ratio=0.5,
#         split_type='uniform'
#     )
#
#     feature_extractor = preprocessing.FeatureExtractor()
#     x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
#     x_test = feature_extractor.transform(x_test)
#
#     # 2. 使用线性核便于解释
#     model = SVM()  # 线性核可解释性强
#     model.fit(x_train, y_train)
#
#     # 3. 获取测试集的详细预测结果
#     test_predictions = model.decision_function(x_test)  # 获取决策值
#     test_results = []
#
#     # 4. 分析关键特征
#     feature_names = feature_extractor.get_feature_names()  # 需要修改FeatureExtractor以支持此方法
#     svm_weights = model.coef_[0]  # 获取特征权重
#
#     # 找到权重最高的特征（最可能指示异常的事件）
#     top_features_idx = np.argsort(np.abs(svm_weights))[::-1][:10]  # 取前10个重要特征
#     top_features = [(feature_names[i], svm_weights[i]) for i in top_features_idx]
#
#     print("\nTop 10 indicative log events:")
#     for feature, weight in top_features:
#         print(f"{feature}: {weight:.4f}")
#
#     # 5. 关联预测结果与原始日志
#     for i in range(len(x_test)):
#         # 获取当前会话窗口的原始日志
#         session_logs = raw_logs['test'][i]
#
#         # 获取异常分数和预测标签
#         anomaly_score = test_predictions[i]
#         is_anomaly = 1 if anomaly_score > 0 else 0
#
#         # 如果是异常会话，分析具体事件
#         if is_anomaly == 1:
#             # 获取该会话中权重最大的事件
#             session_features = x_test[i]
#             top_event_idx = np.argmax(np.abs(session_features * svm_weights))
#             critical_event = feature_names[top_event_idx]
#
#             # 找到原始日志中的关键事件
#             critical_log_entries = [log for log in session_logs if log['EventId'] == critical_event]
#
#             test_results.append({
#                 'session_id': i,
#                 'anomaly_score': anomaly_score,
#                 'critical_event': critical_event,
#                 'critical_count': len(critical_log_entries),
#                 'example_log': critical_log_entries[0]['Content'] if critical_log_entries else "N/A",
#                 'all_events': [log['EventId'] for log in session_logs]
#             })
#
#     # 6. 分析高频异常模式
#     if test_results:
#         anomaly_df = pd.DataFrame(test_results)
#
#         # 统计最常见的异常事件
#         print("\nFrequency of critical events in anomalies:")
#         print(anomaly_df['critical_event'].value_counts().head(5))
#
#         # 保存详细结果
#         anomaly_df.to_csv(r'E:\swarm\swarm\logAgents\data\logs\anomaly_analysis.csv', index=False)
#         return anomaly_df
#
#     return None
#
#
#
#
#
#
# def main():
#     """主函数，用于演示和测试"""
#     anomaly_detect_with_analysis()
#
#
# # 当文件作为独立程序运行时，执行 main 函数
# if __name__ == "__main__":
#     main()

import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from logAgents.configs.tools.detect.loglizer import dataloader, preprocessing
from logAgents.configs.tools.detect.loglizer.models import LogClustering


def anomaly_detect_with_analysis():
    structured_log = r'E:\swarm\swarm\logAgents\data\logs\HDFS_100k.log_structured.csv'
    label_file = r'E:\swarm\swarm\logAgents\data\logs\anomaly_label_100k.csv'
    max_dist = 0.3  # the threshold to stop the clustering process
    anomaly_threshold = 0.3  # the threshold for anomaly detect

    # 1. 加载数据并保留原始日志信息
    (x_train, y_train, train_block_ids), (x_test, y_test, test_block_ids), raw_data = dataloader.load_HDFS_with_raw(
        structured_log,
        label_file=label_file,
        window='session',
        train_ratio=0.5,
        split_type='uniform'
    )

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    # LogClustering 模型
    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    model.fit(x_train[y_train == 0, :])  # Use only normal samples for training


    # 在异常检测阶段把异常日志筛选出来（基于聚类算法）
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # 筛选出异常BlockId
    anomaly_train_blocks = train_block_ids[y_train_pred == 1]
    anomaly_test_blocks = test_block_ids[y_test_pred == 1]
    all_anomaly_blocks = np.concatenate([anomaly_train_blocks, anomaly_test_blocks])

    # 直接从原始日志信息DataFrame中筛选异常日志
    anomaly_details = raw_data[raw_data['BlockId'].isin(all_anomaly_blocks)]

    # # 转换为字典列表便于JSON输出
    # anomaly_details_list = anomaly_details.to_dict('records')
    #
    # # 将anomaly文件保存在本地
    # output_dir = 'detect_result/'
    # os.makedirs(output_dir, exist_ok=True)
    #
    # # JSON文件
    # json_file = os.path.join(output_dir, 'anomaly_logs.json')
    # with open(json_file, 'w', encoding='utf-8') as f:
    #     json.dump(anomaly_details_list, f, ensure_ascii=False, indent=2)
    #
    # # CSV文件
    # csv_file = os.path.join(output_dir, 'anomaly_logs.csv')
    # anomaly_details.to_csv(csv_file, index=False)
    #
    # print(f"发现 {len(all_anomaly_blocks)} 个异常Block，共 {len(anomaly_details)} 条日志记录")
    # print(f"结果已保存至: {json_file} 和 {csv_file}")
    #
    # return anomaly_details

    # 按BlockId整合日志记录
    grouped_anomalies = defaultdict(list)
    for _, row in anomaly_details.iterrows():
        # 提取日志信息
        log_info = {
            'Time': row['Time'],
            'EventId': row['EventId'],
            'Content': row['Content'],
            'EventTemplate': row.get('EventTemplate', ''),
            'Pid': row.get('Pid', ''),
            'Level': row.get('Level', ''),
            # 添加其他可能需要的字段
        }
        grouped_anomalies[row['BlockId']].append(log_info)

    # 创建整合后的数据结构
    consolidated_anomalies = []
    for block_id, logs in grouped_anomalies.items():
        # 按时间戳排序日志
        sorted_logs = sorted(logs, key=lambda x: x['Time'])

        # 获取该block的事件序列
        event_sequence = [log['EventId'] for log in sorted_logs]

        # 创建整合后的条目
        consolidated_entry = {
            'BlockId': block_id,
            'TotalLogs': len(logs),
            'FirstTimestamp': sorted_logs[0]['Time'],
            'LastTimestamp': sorted_logs[-1]['Time'],
            'EventSequence': event_sequence,
            'Logs': sorted_logs,
            # 添加统计信息
            # 'UniqueEvents': len(set(event_sequence)),
            # 'AnomalyScore': np.mean([model.predict_single(log) for log in logs]),  # 如果模型支持单个日志预测
            # 'Label': raw_data[raw_data['BlockId'] == block_id]['Label'].iloc[0] if 'Label' in raw_data else None
        }
        consolidated_anomalies.append(consolidated_entry)

    # 按时间排序（最早出现的异常）
    consolidated_anomalies.sort(key=lambda x: x['FirstTimestamp'])

    # 4. 将整合后的异常文件保存在本地
    output_dir = 'detect_result/'
    os.makedirs(output_dir, exist_ok=True)

    # JSON文件
    json_file = os.path.join(output_dir, 'consolidated_anomalies.json')
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(consolidated_anomalies, f, ensure_ascii=False, indent=2)

    # CSV文件（每个block一行）
    csv_data = []
    for entry in consolidated_anomalies:
        csv_row = {
            'BlockId': entry['BlockId'],
            'TotalLogs': entry['TotalLogs'],
            'FirstTimestamp': entry['FirstTimestamp'],
            'LastTimestamp': entry['LastTimestamp'],
            'EventSequence': ' '.join(entry['EventSequence']),
            # 'UniqueEvents': entry['UniqueEvents'],
            # 'AnomalyScore': entry.get('AnomalyScore', 'N/A'),
            # 'Label': entry.get('Label', 'N/A')
        }
        csv_data.append(csv_row)

    csv_file = os.path.join(output_dir, 'consolidated_anomalies.csv')
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)

    print(f"发现 {len(consolidated_anomalies)} 个异常Block，共 {len(anomaly_details)} 条日志记录")
    print(f"整合结果已保存至: {json_file} 和 {csv_file}")

    return consolidated_anomalies

if __name__ == "__main__":
    anomaly_detect_with_analysis()