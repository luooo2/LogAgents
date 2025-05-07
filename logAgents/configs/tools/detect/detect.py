# from logAgents.configs.tools.detect.loglizer.models import SVM
from logAgents.configs.tools.detect.loglizer.models import LogClustering
from logAgents.configs.tools.detect.loglizer import dataloader, preprocessing

def anomaly_detect(context_variables: dict):
    print("log anomaly detecting……\n")

    structured_log = context_variables.get("structured_log") # The structured log file
    label_file = r'E:\swarm\swarm\logAgents\data\logs\anomaly_label_v1.csv' # The anomaly label file
    max_dist = 0.3  # the threshold to stop the clustering process
    anomaly_threshold = 0.3  # the threshold for anomaly detection

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


    print(context_variables)
    return "日志异常检测完成"
