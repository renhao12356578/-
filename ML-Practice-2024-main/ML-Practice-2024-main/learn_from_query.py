import evaluation_utils as eval_utils
import matplotlib.pyplot as plt
import numpy as np
import range_query as rq
import json
from tqdm import tqdm
import torch
import statistics as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump


def min_max_normalize(v, min_v, max_v):
    # The function may be useful when dealing with lower/upper bounds of columns.
    assert max_v > min_v
    return (v - min_v) / (max_v - min_v)


def get_normalized_val(column_stats, val):
    min_v = column_stats.min_val()
    max_v = column_stats.max_val()
    return min_max_normalize(val, min_v, max_v)


def extract_features_from_query(range_query, table_stats, considered_cols):
    # feat:     [c1_begin, c1_end, c2_begin, c2_end, ... cn_begin, cn_end, AVI_sel, EBO_sel, Min_sel]
    #           <-                   range features                    ->, <-     est features     ->
    # feature = []
    # YOUR CODE HERE: extract features from query

    # 建模：[c1L, c1R, c2L, c2R, …, cNL, cNR]
    feature = [[0, 0] for _ in range(6)]

    for col, val in range_query.col_left.items():
        normalized_val = get_normalized_val(table_stats.columns[col], val)
        col_index = considered_cols.index(col)
        feature[col_index][0] = normalized_val

    for col, val in range_query.col_right.items():
        normalized_val = get_normalized_val(table_stats.columns[col], val)
        col_index = considered_cols.index(col)
        feature[col_index][1] = normalized_val

    return [item for sublist in feature for item in sublist]


def preprocess_queries(queris, table_stats, columns):
    """
    preprocess_queries turn queries into features and labels, which are used for regression model.
    """
    features, labels = [], []
    # queris = queris[:2000]
    for item in queris:
        query, act_rows = item['query'], item['act_rows']
        range_query = rq.ParsedRangeQuery.parse_range_query(query)
        feature = extract_features_from_query(range_query, table_stats, columns)

        # 特征增强
        feature.append(stats.AVIEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        feature.append(stats.ExpBackoffEstimator.estimate(range_query, table_stats) * table_stats.row_count)
        feature.append(stats.MinSelEstimator.estimate(range_query, table_stats) * table_stats.row_count)

        features.append(feature)
        labels.append(act_rows)
    return features, labels


class QueryDataset(torch.utils.data.Dataset):
    def __init__(self, queries, table_stats, columns):
        super().__init__()
        self.query_data = list(zip(*preprocess_queries(queries, table_stats, columns)))
        self.features, self.labels = preprocess_queries(queries, table_stats, columns)

    def __getitem__(self, index):
        return self.query_data[index]
        # feature = torch.tensor(self.features[index], dtype=torch.float32)
        # label = torch.tensor(self.labels[index], dtype=torch.float32)
        # return feature, label

    def __len__(self):
        return len(self.query_data)


def calculate_q_error(actual, predicted):
    """
    Calculate q-error for a list of actual and predicted values.
    """
    q_errors = [
        max(act / est, est / act) if act > 0 and est > 0 else float('inf')  # Avoid division by zero
        for act, est in zip(actual, predicted)
    ]
    return q_errors


def load_data(train_data, test_data, table_stats, columns):
    train_dataset = QueryDataset(train_data, table_stats, columns)
    test_dataset = QueryDataset(test_data, table_stats, columns)
    return train_dataset, test_dataset


# LogisticRegression
def est_AI1(train_dataset, test_dataset, model_saved_path):
    """
    Produce estimated rows for train_data and test_data using Logistic Regression with train_loader and test_loader.
    """
    # Prepare training data from train_loader
    train_features, train_labels = [], []
    for data in train_dataset:
        features, label = data
        train_features.append(features)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print("LogisticRegression training start")
    # Define and train the model
    # model = LogisticRegression(max_iter=1000,n_jobs=-1)
    # model.fit(train_features, train_labels)

    model = LogisticRegression(max_iter=1, n_jobs=-1)
    for epoch in tqdm(range(100), desc="LogisticRegression Training Progress"):
        model.fit(train_features, train_labels)

    # Prepare testing data from test_loader
    test_features, test_labels = [], []
    for data in test_dataset:
        features, label = data
        test_features.append(features)
        test_labels.append(label)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Make predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate q-errors for train and test
    train_q_errors = calculate_q_error(train_labels, train_predictions)
    test_q_errors = calculate_q_error(test_labels, test_predictions)

    # Calculate MSE of q-errors
    train_mse = np.mean(np.square(train_q_errors))
    test_mse = np.mean(np.square(test_q_errors))

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    dump(model, model_saved_path + '/LogisticRegression.joblib')
    print("model saved!")

    # Convert results to lists for output
    train_est_rows = train_predictions.tolist()
    train_act_rows = train_labels.tolist()
    test_est_rows = test_predictions.tolist()
    test_act_rows = test_labels.tolist()

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


# RandomForest
def est_AI2(train_dataset, test_dataset, model_saved_path):
    """
    produce estimated rows for train_data and test_data
    """
    # Prepare training data from train_loader
    train_features, train_labels = [], []
    for data in train_dataset:
        features, label = data
        train_features.append(features)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print("RandomForest training start")
    # Define and train the model
    # model = LogisticRegression(max_iter=1000,n_jobs=-1)
    # model.fit(train_features, train_labels)

    model = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    for epoch in tqdm(range(1000), desc="RandomForest Training Progress"):
        model.fit(train_features, train_labels)

    # Prepare testing data from test_loader
    test_features, test_labels = [], []
    for data in test_dataset:
        features, label = data
        test_features.append(features)
        test_labels.append(label)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Make predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate q-errors for train and test
    train_q_errors = calculate_q_error(train_labels, train_predictions)
    test_q_errors = calculate_q_error(test_labels, test_predictions)

    # Calculate MSE of q-errors
    train_mse = np.mean(np.square(train_q_errors))
    test_mse = np.mean(np.square(test_q_errors))

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    dump(model, model_saved_path + '/RandomForest.joblib')
    print("model saved!")

    # Convert results to lists for output
    train_est_rows = train_predictions.tolist()
    train_act_rows = train_labels.tolist()
    test_est_rows = test_predictions.tolist()
    test_act_rows = test_labels.tolist()

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


# KNN
def est_AI3(train_dataset, test_dataset, model_saved_path):
    """
    produce estimated rows for train_data and test_data
    """
    # Prepare train and test loaders
    # Prepare training data from train_loader
    train_features, train_labels = [], []
    for data in train_dataset:
        features, label = data
        train_features.append(features)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print("KNN training start")
    # Define and train the model
    # model = LogisticRegression(max_iter=1000,n_jobs=-1)
    # model.fit(train_features, train_labels)

    model = KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    model.fit(train_features, train_labels)

    # Prepare testing data from test_loader
    test_features, test_labels = [], []
    for data in test_dataset:
        features, label = data
        test_features.append(features)
        test_labels.append(label)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Make predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate q-errors for train and test
    train_q_errors = calculate_q_error(train_labels, train_predictions)
    test_q_errors = calculate_q_error(test_labels, test_predictions)

    # Calculate MSE of q-errors
    train_mse = np.mean(np.square(train_q_errors))
    test_mse = np.mean(np.square(test_q_errors))

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    dump(model, model_saved_path + '/KNN.joblib')
    print("model saved!")

    # Convert results to lists for output
    train_est_rows = train_predictions.tolist()
    train_act_rows = train_labels.tolist()
    test_est_rows = test_predictions.tolist()
    test_act_rows = test_labels.tolist()

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


# SVM
def est_AI4(train_dataset, test_dataset, model_saved_path):
    """
    produce estimated rows for train_data and test_data
    """
    # Prepare training data from train_loader
    train_features, train_labels = [], []
    for data in train_dataset:
        features, label = data
        train_features.append(features)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print("SVM training start")
    # Define and train the model
    # model = LogisticRegression(max_iter=1000,n_jobs=-1)
    # model.fit(train_features, train_labels)

    model = LinearSVR(max_iter=1000, random_state=42)
    model.fit(train_features, train_labels)

    # Prepare testing data from test_loader
    test_features, test_labels = [], []
    for data in test_dataset:
        features, label = data
        test_features.append(features)
        test_labels.append(label)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Make predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate q-errors for train and test
    train_q_errors = calculate_q_error(train_labels, train_predictions)
    test_q_errors = calculate_q_error(test_labels, test_predictions)

    # Calculate MSE of q-errors
    train_mse = np.mean(np.square(train_q_errors))
    test_mse = np.mean(np.square(test_q_errors))

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    dump(model, model_saved_path + '/LinearSVR.joblib')
    print("model saved!")

    # Convert results to lists for output
    train_est_rows = train_predictions.tolist()
    train_act_rows = train_labels.tolist()
    test_est_rows = test_predictions.tolist()
    test_act_rows = test_labels.tolist()

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


# LightBGM
def est_AI5(train_dataset, test_dataset, model_saved_path):
    """
    produce estimated rows for train_data and test_data
    """
    # Prepare training data from train_loader
    train_features, train_labels = [], []
    for data in train_dataset:
        features, label = data
        train_features.append(features)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print("LightBGM training start")
    # Define and train the model
    # model = LogisticRegression(max_iter=1000,n_jobs=-1)
    # model.fit(train_features, train_labels)

    model = LGBMRegressor(
        n_estimators=200,  # 使用 200 棵树
        max_depth=-1,  # 自动调整深度
        learning_rate=0.1,  # 学习率
        random_state=42,  # 随机种子
        n_jobs=-1  # 多线程支持
    )

    model.fit(train_features, train_labels)

    # Prepare testing data from test_loader
    test_features, test_labels = [], []
    for data in test_dataset:
        features, label = data
        test_features.append(features)
        test_labels.append(label)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Make predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate q-errors for train and test
    train_q_errors = calculate_q_error(train_labels, train_predictions)
    test_q_errors = calculate_q_error(test_labels, test_predictions)

    # Calculate MSE of q-errors
    train_mse = np.mean(np.square(train_q_errors))
    test_mse = np.mean(np.square(test_q_errors))

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    dump(model, model_saved_path + '/LightBGM.joblib')
    print("model saved!")

    # Convert results to lists for output
    train_est_rows = train_predictions.tolist()
    train_act_rows = train_labels.tolist()
    test_est_rows = test_predictions.tolist()
    test_act_rows = test_labels.tolist()

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


# XGBoost
def est_AI6(train_dataset, test_dataset, model_saved_path):
    """
    produce estimated rows for train_data and test_data
    """
    # Prepare training data from train_loader
    train_features, train_labels = [], []
    for data in train_dataset:
        features, label = data
        train_features.append(features)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print("XGBoost training start")
    # Define and train the model
    # model = LogisticRegression(max_iter=1000,n_jobs=-1)
    # model.fit(train_features, train_labels)

    model = XGBRegressor(
        n_estimators=200,  # 树的数量
        max_depth=6,  # 每棵树的最大深度
        learning_rate=0.1,  # 学习率
        random_state=42,  # 随机种子
        n_jobs=-1  # 多线程支持
    )

    model.fit(train_features, train_labels)

    # Prepare testing data from test_loader
    test_features, test_labels = [], []
    for data in test_dataset:
        features, label = data
        test_features.append(features)
        test_labels.append(label)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Make predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate q-errors for train and test
    train_q_errors = calculate_q_error(train_labels, train_predictions)
    test_q_errors = calculate_q_error(test_labels, test_predictions)

    # Calculate MSE of q-errors
    train_mse = np.mean(np.square(train_q_errors))
    test_mse = np.mean(np.square(test_q_errors))

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    dump(model, model_saved_path + '/LightBGM.joblib')
    print("model saved!")

    # Convert results to lists for output
    train_est_rows = train_predictions.tolist()
    train_act_rows = train_labels.tolist()
    test_est_rows = test_predictions.tolist()
    test_act_rows = test_labels.tolist()

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


# MLP
def est_AI7(train_dataset, test_dataset, model_saved_path):
    """
    produce estimated rows for train_data and test_data
    """
    # Prepare training data from train_loader
    train_features, train_labels = [], []
    for data in train_dataset:
        features, label = data
        train_features.append(features)
        train_labels.append(label)

    train_features = np.array(train_features)
    train_labels = np.array(train_labels)

    print("MLP training start")
    # Define and train the model
    # model = LogisticRegression(max_iter=1000,n_jobs=-1)
    # model.fit(train_features, train_labels)

    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),  # 两层隐藏层
        max_iter=1000,  # 最大迭代次数
        learning_rate='adaptive',  # 动态调整学习率
        random_state=42,  # 设置随机种子
        early_stopping=True  # 启用早停以防止过拟合
    )

    model.fit(train_features, train_labels)

    # Prepare testing data from test_loader
    test_features, test_labels = [], []
    for data in test_dataset:
        features, label = data
        test_features.append(features)
        test_labels.append(label)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    # Make predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate q-errors for train and test
    train_q_errors = calculate_q_error(train_labels, train_predictions)
    test_q_errors = calculate_q_error(test_labels, test_predictions)

    # Calculate MSE of q-errors
    train_mse = np.mean(np.square(train_q_errors))
    test_mse = np.mean(np.square(test_q_errors))

    print(f"Train MSE: {train_mse}")
    print(f"Test MSE: {test_mse}")

    dump(model, model_saved_path + '/MLP.joblib')
    print("model saved!")

    # Convert results to lists for output
    train_est_rows = train_predictions.tolist()
    train_act_rows = train_labels.tolist()
    test_est_rows = test_predictions.tolist()
    test_act_rows = test_labels.tolist()

    return train_est_rows, train_act_rows, test_est_rows, test_act_rows


def eval_model(model, train_data, test_data, table_stats, columns):
    if model == 'ai1':
        est_fn = est_AI1
    else:
        est_fn = est_AI2

    train_est_rows, train_act_rows, test_est_rows, test_act_rows = est_fn(train_data, test_data, table_stats, columns)

    name = f'{model}_train_{len(train_data)}'
    eval_utils.draw_act_est_figure(name, train_act_rows, train_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(train_act_rows, train_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')

    name = f'{model}_test_{len(test_data)}'
    eval_utils.draw_act_est_figure(name, test_act_rows, test_est_rows)
    p50, p80, p90, p99 = eval_utils.cal_p_error_distribution(test_act_rows, test_est_rows)
    print(f'{name}, p50:{p50}, p80:{p80}, p90:{p90}, p99:{p99}')


if __name__ == '__main__':
    stats_json_file = 'data/title_stats.json'
    train_json_file = 'data/query_train_18000.json'
    test_json_file = 'data/validation_2000.json'
    columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
    table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
    with open(train_json_file, 'r') as f:
        train_data = json.load(f)
    with open(test_json_file, 'r') as f:
        test_data = json.load(f)

    eval_model('your_ai_model1', train_data, test_data, table_stats, columns)
    eval_model('your_ai_model2', train_data, test_data, table_stats, columns)
