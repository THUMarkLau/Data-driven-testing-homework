import os

from scipy.io import loadmat
from constant import DATASET_NAMES, TRAIN_PATHS, TEST_PATHS
from sklearn.metrics import confusion_matrix, f1_score


def load_data(path):
    key = os.path.split(path)[1].split(".")[0]
    data = loadmat(path)
    return [data[key][0][i] for i in range(3)]


def load_train_and_test_data():
    train_data = {}
    test_data = {}
    for i in range(len(DATASET_NAMES)):
        train_data[DATASET_NAMES[i]] = load_data(TRAIN_PATHS[i])
        test_data[DATASET_NAMES[i]] = load_data(TEST_PATHS[i])
    return train_data, test_data


def train_and_valid(model, train_data, test_data):
    train_x = train_data[:, :-2]
    train_y = train_data[:, -1]
    model.fit(train_x, train_y)

    test_x = test_data[:, :-2]
    test_y = test_data[:, -1]
    model_pred = model.predict(test_x)

    return estimate(test_y, model_pred)


def estimate(test_y, pred_y):
    matrix = confusion_matrix(y_true=test_y, y_pred=pred_y)
    return {
        'precision': round(matrix[1, 1] / (matrix[0, 1] + matrix[1, 1]), 3),
        'recall': round(matrix[1, 1] / (matrix[1, 0] + matrix[1, 1]), 3),
        'pf': round(matrix[0, 1] / (matrix[0, 0] + matrix[0, 1]), 3),
        'f1': round(f1_score(y_true=test_y, y_pred=pred_y), 3)
    }
