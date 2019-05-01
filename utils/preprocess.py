import ujson as json
from functools import reduce
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from scipy import sparse
from stanfordcorenlp import StanfordCoreNLP
from nltk.corpus import stopwords
from flair.embeddings import ELMoEmbeddings
import pickle

import GraphGen as graph
import Embedder as emb

def read_questions(fname):
    f = open(fname, 'r')
    data = []
    for line in f.readlines():
        data.append(json.loads(line))
    f.close()
    return data

def read_corpus():
    f = open(corpus_file, 'r')
    data = list(f.readlines())
    return data

def tokenize_questions_pos_tagging(data):
    pos_tags = ['NN', 'RB', 'VB']
    for qdata in tqdm(data):
        ques = qdata['question']['stem']
        choices = [c['text'] for c in qdata['question']['choices']]
        ques_pos = nlp.pos_tag(ques)
        tagged_tokens = set()
        for entity in ques_pos:
            if any(entity[1].startswith(tt) for tt in pos_tags):
                tagged_tokens.add(entity[0])
        for choice in qdata['question']['choices']:
            choice_pos = nlp.pos_tag(choice['text'])
            for entity in choice_pos:
                if any(entity[1].startswith(tt) for tt in pos_tags):
                    tagged_tokens.add(entity[0])
            if choice['label'] == '1': choice['label'] = 'A'
            elif choice['label'] == '2': choice['label'] = 'B'
            elif choice['label'] == '3': choice['label'] = 'C'
            elif choice['label'] == '4': choice['label'] = 'D'
        qdata['tagged_tokens'] = list(tagged_tokens)
    return data

def aggragate_tokens(data):
    ts = reduce(lambda x, y: x.union(y), [set(qdata['tagged_tokens']) for qdata in data])
    ts = set(ts)
    token_to_id = {t: i for i, t in enumerate(ts)}
    return token_to_id

def get_vectorizer(token_to_id):
    return CountVectorizer(vocabulary=token_to_id)

def vectorize_corpus(corpus, vectorizer):
    vec = vectorizer.transform(corpus)
    vec[vec>=1]=1
    return vec

def compute_unigrams_from_vectors(corpus_vec):
    unigrams = np.squeeze(np.asarray(np.sum(corpus_vec, axis=0)))
    unigrams = unigrams / np.sum(unigrams)
    unigrams[unigrams == 0] = 1
    unigrams = -1 * np.log(unigrams)
    return unigrams

def find_neighbors_of_questions(data, corpus_vec, weights, vectorizer, n_neighbors=500, step_size=10):
    idx = 0
    for qdata in tqdm(range(0, len(data), step_size)):
        q = vectorizer.transform([' '.join(qdata['tagged_tokens']) for qdata in data[idx:idx+step_size]])
        q[q>=1] = 1
        wq = np.asarray(q.todense()) * weights
        mult = sparse.csr_matrix.dot(wq, corpus_vec.T)
        idcs = mult.argsort()[:, -n_neighbors:].tolist()
        for i, val in enumerate(idcs):
            data[idx+i]['neighbors'] = val[::-1]
        idx += step_size
    return data

def store_final_data(fname, data):
    f = open('%s.pkl' % fname, 'wb')
    pickle.dump(data, f)
    f.close()

if __name__ == '__main__':
    nlp = StanfordCoreNLP('../stanford-corenlp-full-2018-10-05')

    corpus_file = './ARC-V1-Feb2018-2/ARC_Corpus.txt'
    questions_file = './ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Train.jsonl'
    dev_questions_file = './ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Dev.jsonl'
    top_k_neighbors = 100
    depth = 5
    tags_to_remove = ['aux', 'auxpass', 'cc', 'ccomp', 'conj', 'csubj', 'case', 'csubjpass', 'det', 'discourse', 'expl', 'goeswith', 'mwe', 'parataxis', 'pcomp', 'pobj', 'preconj', 'predet', 'prep', 'prepc', 'prt', 'punct', 'quantmod', 'root', 'vmod', 'acomp']
    stop_words = [word for word in stopwords.words('english')]
    dep_type = 'basicDependencies'
    output_file_name = 'ARC-Final'
    embedding_model = 'small'
    embedding_dim = 768

    data = read_questions(questions_file)
    dev_data = read_questions(dev_questions_file)
    data = tokenize_questions_pos_tagging(data)
    dev_data = tokenize_questions_pos_tagging(dev_data)
    token_to_id = aggragate_tokens(data)
    corpus = read_corpus()
    cv = get_vectorizer(token_to_id)
    corpus_vec = vectorize_corpus(corpus, cv)

    unigrams = compute_unigrams_from_vectors(corpus_vec)
    data = find_neighbors_of_questions(data, corpus_vec, unigrams, cv)
    dev_data = find_neighbors_of_questions(dev_data, corpus_vec, unigrams, cv)
    total = set(neighbor for qdata in data for neighbor in qdata['neighbors'][:top_k_neighbors])
    dev_total = set(neighbor for qdata in dev_data for neighbor in qdata['neighbor'][:top_k_neighbors])
    total = total.union(dev_total)
    relevant_sentences = sorted(list(total))

    dep_props = {'annotators': 'depparse', 'pipelineLanguage': 'en'}
    depparsed = {}
    for sentence_idx in tqdm(relevant_sentences):
        depparsed[sentence_idx] = [sent[dep_type] for sent in json.loads(nlp.annotate(corpus[sentence_idx], properties=dep_props))['sentences']]

    annotator = lambda x: json.loads(nlp.annotate(x, properties=dep_props))
    for qdata in tqdm(data):
        qdata['edges'] = graph.get_edge_list(qdata, depparse, annotator, depth=depth, n_neighbors=top_k_neighbors, dep_type=dep_type)

    for qdata in tqdm(data):
        graph.prune_graph(qdata, tags_to_remove, stop_words)

    for qdata in tqdm(data):
        sentential_nodes_to_idx, doc_neighbors, match_neighbors = graph.get_final_graph_representation(qdata, directed=False)
        qdata['sentential_nodes_to_idx'] = sentential_nodes_to_idx
        qdata['doc_neighbors'] = doc_neighbors
        qdata['match_neighbors'] = match_neighbors
        qdata['question_idx'] = set([v for k,v in qdata['sentential_nodes_to_idx'].items() if k[0] == 'question'])
        for ch in qdata['question']['choices']:
            qdata['choice_%s_idx' % ch['label']] = set([v for k,v in qdata['sentential_nodes_to_idx'].items() if k[0] == 'choice:%s' % ch['label']])
        del qdata['edges']

    tokenizer = lambda x: nlp.word_tokenize(x)
    embedder = ELMoEmbeddings(embedding_model)
    neighbor_indices = list(set(n[0] for qdata in data for n in qdata['sentential_nodes_to_idx'] if type(n[0]) == int))
    outputs = emb.get_corpus_embeddings(corpus, neighbor_indices, embedder, tokenizer)
    reverse_neighbor_idx_mapping = {v: i for i, v in enumerate(neighbor_indices)}
    for qdata in tqdm(data):
        qdata['node_embedding_matrix'] = emb.get_question_embedding_matrix(qdata, outputs, reverse_neighbor_idx_mapping, embedding_dim)

    store_final_data(output_file_name, qdata)
