import numpy as np
from tqdm import tqdm

def embed_sentences(sentences, embedder, tokenizer):
    outputs = []
    tokenized_sentences = [tokenizer(s) for s in sentences]
    emb = embedder.ee.embed_batch(tokenized_sentences)
    for sidx in range(len(tokenized_sentences)):
        outputs.append(np.concatenate([emb[sidx][0], emb[sidx][1], emb[sidx][2]], axis=1))
    return outputs

def get_corpus_embeddings(corpus, neighbor_indices, embedder, tokenizer):
    outputs = []
    batch_size = 10
    for i in tqdm(range(0, len(neighbor_indices)//batch_size)):
        idcs = neighbor_indices[i*batch_size:(i+1)*batch_size]
        sentences = [corpus[j] for j in idcs]
        outputs.extend(embed_sentences(sentences, embedder, tokenizer))

    last_idx = len(neighbor_indices)//batch_size * batch_size
    idcs = neighbor_indices[last_idx:]
    sentences = [corpus[s] for s in idcs]
    outputs.extend(embed_sentences(sentences, embedder, tokenizer))
    return outputs

def get_question_embedding_matrix(qdata, outputs, reverse_neighbor_idx_mapping, embedding_dim=768):
    nodes_embedding_matrix = np.zeros((len(qdata['sentential_nodes_to_idx']), embedding_dim))
    qembedding = embed_sentences([qdata['question']['stem']])[0]
    choices = {c['label']: c['text'] for c in qdata['question']['choices']}
    for n, idx in qdata['sentential_nodes_to_idx'].items():
        if type(n[0]) == int:
            nidx = reverse_neighbor_idx_mapping[n[0]]
            nodes_embedding_matrix[idx] = outputs[nidx][n[1] - 1]
        elif n[0] == 'question':
            nodes_embedding_matrix[idx] = qembedding[n[1] - 1]
        elif n[0].startswith('choice'):
            cembedding = embed_sentences([choices[n[0].split(':')[1]]])[0]
            nodes_embedding_matrix[idx] = cembedding[n[1] - 1]
    return nodes_embedding_matrix
