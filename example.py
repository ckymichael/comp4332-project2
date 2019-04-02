import random
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas
from tqdm import tqdm


def get_neighbourhood_score(local_model, node1, node2):
    # Provide the plausibility score for a pair of nodes based on your own model.


def get_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)




# Start to load the train data

train_edges = list()
raw_train_data = pandas.read_csv('train.csv')
for i, record in raw_train_data.iterrows():
    train_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the train data.')

# Start to load the valid/test data

valid_positive_edges = list()
valid_negative_edges = list()
raw_valid_data = pandas.read_csv('valid.csv')
for i, record in raw_valid_data.iterrows():
    if record['label']:
        valid_positive_edges.append((str(record['head']), str(record['tail'])))
    else:
        valid_negative_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the valid/test data.')


# write code to train the model here


# replace 'your_model' with your own model and use the provided evaluation code to evaluate.
tmp_AUC_score = get_AUC(Your_model, valid_positive_edges, valid_negative_edges)

print('tmp_accuracy:', tmp_AUC_score)

print('end')
