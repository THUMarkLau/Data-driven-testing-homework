from utils import load_train_and_test_data, train_and_valid
from constant import MODELS, DATASET_NAMES

if __name__ == '__main__':
    train_data, test_data = load_train_and_test_data()
    for dataset in DATASET_NAMES:
        cur_dataset_for_train = train_data[dataset]
        cur_dataset_for_test = test_data[dataset]
        for model in MODELS:
            print("training " + model[0] + " in " + dataset)
            for i in range(3):
                print(train_and_valid(model[1], cur_dataset_for_train[i], cur_dataset_for_test[i]))
