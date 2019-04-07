import random
import numpy as np
import logging
import networkx as nx
from sklearn.metrics import roc_auc_score
import pandas
from tqdm import tqdm
from deepwalk import graph
from deepwalk import walks as serialized_walks
from deepwalk import skipgram
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def load_csv(file_, undirected=True):
    G = graph.Graph()
    raw_data = pandas.read_csv(file_)
    for i, record in raw_data.iterrows():
        G[str(record['head'])].append(str(record['tail']))
        if undirected:
             G[str(record['tail'])].append(str(record['head']))
    G.make_consistent()
    return G

def get_neighbourhood_score(local_model, node1, node2):
    try:
        vector1 = local_model.wv.syn0[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.syn0[local_model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        return random.random()

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

def predict(model, edges):
    prediction_list = list()
    for edge in edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        prediction_list.append(tmp_score)
    y_scores = np.array(prediction_list)
    return y_scores

# Start to load the train data
G = load_csv('../data/train.csv')
print('finish loading the train data.')

# Start to load the valid/test data

valid_positive_edges = list()
valid_negative_edges = list()
raw_valid_data = pandas.read_csv('../data/valid.csv')
for i, record in raw_valid_data.iterrows():
    if record['label']:
        valid_positive_edges.append((str(record['head']), str(record['tail'])))
    else:
        valid_negative_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the valid/test data.')

# hyperparameter
number_walks = 5
walk_length = 10
dimension = 15
window_size = 5
workers = 10
iterations = 20

print("Number of nodes: {}".format(len(G.nodes())))
num_walks = len(G.nodes()) * number_walks
print("Number of walks: {}".format(num_walks))
data_size = num_walks * walk_length
print("Data size (walks*length): {}".format(data_size))

print("Walking...")
walks = graph.build_deepwalk_corpus(G, num_paths= number_walks,path_length=walk_length, alpha=0, rand=random.Random(0))

print("Training...")
model = Word2Vec(walks, size=dimension, window=window_size, min_count=0, sg=1, hs=1,workers=workers)

# save resulted_embeddings 
resulted_embeddings = dict()
for i, w in enumerate(model.wv.index2word):
    resulted_embeddings[w] = model.wv.syn0[i]

# AUC-ROC score on the validation set
tmp_AUC_score = get_AUC(model, valid_positive_edges, valid_negative_edges)
print('tmp_accuracy:', tmp_AUC_score)

# test set prediction
#test_edges = list()
#raw_test_data = pandas.read_csv('../data/test.csv')
#for i, record in raw_test_data.iterrows():
#    test_edges.append((str(record['head']), str(record['tail'])))

#test_predict = predict(model,test_edges)

#df = pandas.DataFrame({'head':raw_test_data['head'],'label': 'N/A','score':test_predict,'tail':raw_test_data['tail']})
#df.to_csv("../test.csv",index  = 0)

print('end')



