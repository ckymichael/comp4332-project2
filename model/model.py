import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from random import random

import node2vec
from networkx import DiGraph
from gensim.models import Word2Vec

from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def get_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model.wv.vectors[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.vectors[local_model.wv.index2word.index(node2)]
        res = 1 - cosine(vector1, vector2)
    except ValueError:
        res = random()

    return res


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


def get_G_from_edges(edges):
    edge_dict = dict()
    tmp_G = DiGraph()

    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1

    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        tmp_G.add_edge(edge_key.split('_')[0], edge_key.split('_')[1])
        tmp_G[edge_key.split('_')[0]][edge_key.split('_')[1]]['weight'] = weight

    return tmp_G


def load_edges(data):
    edges = []
    for i in range(data.shape[0]):
        head = data.at[i, 'head']
        tail = data.at[i, 'tail']
        edges.append((head, tail))

    return edges


def build_pred(model):
    print('Predicting test data')
    test_data = pd.read_csv('data/test.csv', index_col=0)
    test_edges = load_edges(test_data)

    predictions = []
    for edge in tqdm(test_edges):
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        predictions.append(tmp_score)

    test_data['score'] = pd.Series(predictions)

    print('Outpting predictions')
    test_data.to_csv('pred.csv', index=False)


train_data = pd.read_csv('data/train.csv', index_col=0)
train_edges = load_edges(train_data)

print('Finish loading training data.')

valid_positive_edges = list()
valid_negative_edges = list()
valid_data = pd.read_csv('data/valid.csv', index_col=0)

for i in range(valid_data.shape[0]):
    head = valid_data.loc[i, 'head']
    tail = valid_data.loc[i, 'tail']
    if valid_data.loc[i, 'label']:
        valid_positive_edges.append((head, tail))
    else:
        valid_negative_edges.append((head, tail))

print('Finish loading validation data.')

directed = False
p = 1
q = 1
num_walks = 16
walk_length = 8
dimension = 8
window_size = 4
num_workers = 8
iterations = 16

G = node2vec.Graph(get_G_from_edges(train_edges), directed, p, q)
G.preprocess_transition_probs()
walks = G.simulate_walks(num_walks, walk_length)

model = Word2Vec(walks, size=dimension, window=window_size, min_count=0, sg=1, workers=num_workers, iter=iterations)
resulted_embeddings = dict()
for i, w in enumerate(model.wv.index2word):
    resulted_embeddings[w] = model.wv.vectors[i]

tmp_AUC_score = get_AUC(model, valid_positive_edges, valid_negative_edges)

print('AUC Score (Validation): %.6f' % tmp_AUC_score)

build_pred(model)

print('End of model')
