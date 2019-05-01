import numpy as np
import pickle

def load_final_data():
    f = open('OpenBookQA_train_d5_Questions_with_embeddings.pkl', 'rb')
    data = pickle.load(f)
    f.close()
    return data

def load_dev_data():
    f = open('OpenBookQA_dev_d5_Questions_with_embeddings.pkl', 'rb')
    data = pickle.load(f)
    f.close()
    return data

def generate_adj_matrix(qdata):
    doc_edges = [(int(n), int(v)) for n in qdata['doc_neighbors'] for v in qdata['doc_neighbors'][n]]
    match_edges = [(int(n), int(v)) for n in qdata['match_neighbors'] for v in qdata['match_neighbors'][n]]
    k = len(qdata['sentential_nodes_to_idx'])
    doc_adj = np.zeros((k, k))
    adj_u = [u for (u, _) in doc_edges]
    adj_v = [v for (_, v) in doc_edges]
    doc_adj[adj_u, adj_v] = 1
    adj_u = [u for (u, _) in match_edges]
    adj_v = [v for (_, v) in match_edges]
    match_adj = np.zeros((k, k))
    match_adj[adj_u, adj_v] = 1
    return doc_adj, match_adj

def get_graph_inputs(qdata):
    de, me = generate_adj_matrix(qdata)
    de = normalize_adj_mat(de)
    me = normalize_adj_mat(me)
    features = qdata['node_embedding_matrix']
    return de, me, features

def normalize_adj_mat(adj_mat):
    rowsum = np.array(adj_mat.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return adj_mat.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
