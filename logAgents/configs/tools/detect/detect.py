# from logAgents.configs.tools.detect.loglizer.models import SVM
import json

import numpy as np

from logAgents.configs.tools.detect.loglizer.models import LogClustering
from logAgents.configs.tools.detect.loglizer import dataloader, preprocessing

def anomaly_detect(context_variables: dict):
    print("log anomaly detecting……\n")

    structured_log = context_variables.get("structured_log") # The structured log file
    label_file = r'E:\swarm\swarm\logAgents\data\logs\anomaly_label_100k.csv' # The anomaly label file
    max_dist = 0.3  # the threshold to stop the clustering process
    anomaly_threshold = 0.3  # the threshold for anomaly detect
    (x_train, y_train), (x_test, y_test) = dataloader.load_HDFS(structured_log,
                                                                label_file=label_file,
                                                                window='session',
                                                                train_ratio=0.5,
                                                                split_type='uniform')

    feature_extractor = preprocessing.FeatureExtractor()
    x_train = feature_extractor.fit_transform(x_train, term_weighting='tf-idf')
    x_test = feature_extractor.transform(x_test)

    # SVM 模型
    # model = SVM()
    # model.fit(x_train, y_train)

    # LogClustering 模型
    model = LogClustering(max_dist=max_dist, anomaly_threshold=anomaly_threshold)
    model.fit(x_train[y_train == 0, :])  # Use only normal samples for training


    print('Train validation:')
    precision, recall, f1 = model.evaluate(x_train, y_train)

    print('Test validation:')
    precision, recall, f1 = model.evaluate(x_test, y_test)


    # 在异常检测阶段把异常日志筛选出来（基于聚类算法）
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # 筛选出异常日志
    anomaly_train_indices = np.where(y_train_pred == 1)[0]
    anomaly_train_logs = [x_train[i] for i in anomaly_train_indices]

    anomaly_test_indices = np.where(y_test_pred == 1)[0]
    anomaly_test_logs = [x_test[i] for i in anomaly_test_indices]

    # 记录每个异常日志的详细信息
    anomaly_details = []
    for idx in anomaly_train_indices:
        log_vector = x_train[idx]
        # min_dist, closest_cluster_id = model._get_min_cluster_dist(log_vector)

        anomaly_details.append({
            "log_id": idx,
            "log_content": log_vector,  # 获取原始日志文本
            # "anomaly_score": min_dist,  # 异常分数（距离最近簇的距离）
            # "closest_cluster": closest_cluster_id,  # 最近正常簇ID
            # "cluster_center": model.representatives[closest_cluster_id],  # 最近簇中心
            # "deviation": log_vector - model.representatives[closest_cluster_id]  # 差异向量
        })
    
    # 将anomaly文件保存在本地
    output_file = 'detect_result/'
    file_name = output_file + 'anomaly_logs.json'
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(anomaly_details, f, ensure_ascii=False, indent=2)




    print(context_variables)
    return "日志异常检测完成"

# def save_to_json(self, filename=None):
#     """保存为JSON格式"""
#     if not filename:
#         filename = f"anomaly_details_{self.timestamp}.json"
#
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(self.anomaly_details, f, ensure_ascii=False, indent=2)
#
#     print(f"已保存 {len(self.anomaly_details)} 条异常记录到 {filename}")
#     return filename

# def save_to_csv(self, filename=None):
#     """保存为CSV格式"""
#     if not filename:
#         filename = f"anomaly_details_{self.timestamp}.csv"
#
#     # 创建扁平化的数据结构用于CSV
#     csv_data = []
#     for record in self.anomaly_details:
#         flat_record = {
#             "log_id": record["log_id"],
#             "timestamp": record["timestamp"],
#             "log_content": record["log_content"],
#             "anomaly_score": record["anomaly_score"],
#             "closest_cluster": record["closest_cluster"],
#             "cluster_size": record["cluster_size"],
#             "top_feature_1_index": record["top_mismatch_features"][0][0],
#             "top_feature_1_value": record["top_mismatch_features"][0][1],
#             "top_feature_1_cluster": record["top_mismatch_features"][0][2],
#             "top_feature_1_deviation": record["top_mismatch_features"][0][3],
#             "top_feature_2_index": record["top_mismatch_features"][1][0],
#             "top_feature_2_value": record["top_mismatch_features"][1][1],
#             "top_feature_2_cluster": record["top_mismatch_features"][1][2],
#             "top_feature_2_deviation": record["top_mismatch_features"][1][3],
#             "top_feature_3_index": record["top_mismatch_features"][2][0],
#             "top_feature_3_value": record["top_mismatch_features"][2][1],
#             "top_feature_3_cluster": record["top_mismatch_features"][2][2],
#             "top_feature_3_deviation": record["top_mismatch_features"][2][3],
#         }
#         csv_data.append(flat_record)
#
#     # 写入CSV文件
#     with open(filename, 'w', newline='', encoding='utf-8') as f:
#         writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
#         writer.writeheader()
#         writer.writerows(csv_data)
#
#     print(f"已保存 {len(self.anomaly_details)} 条异常记录到 {filename}")
#     return filename