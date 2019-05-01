from tqdm.autonotebook import tqdm
import numpy as np
from flair.embeddings import ELMoEmbeddings
from stanfordcorenlp import StanfordCoreNLP
import torch

import utils

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class BatchGenerator(object):
    def __init__(self, data, preprocessed=False):
        self.data = data
        self._cursor = 0
        self.embedder = ELMoEmbeddings('small')
        self.nlp = StanfordCoreNLP('../stanford-corenlp-full-2018-10-05')
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.key_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        self.idcs = list(range(len(self.data)))
        self.shuffle()
        if not preprocessed:
            self.preprocess()

    def preprocess_qdata(self, qdata):
        de, me, x = utils.get_graph_inputs(qdata)
        qembedding = self.get_embedding(qdata['question']['stem'])
        qnodes_idx = list(set(v for n, v in qdata['sentential_nodes_to_idx'].items() if type(n[0]) == str and n[0] == 'question'))
        choices_nodes_idx = {}
        for ch in [c['label'] for c in qdata['question']['choices']]:
            choices_nodes_idx[ch] = [v for n, v in qdata['sentential_nodes_to_idx'].items() if type(n[0]) == str and n[0] == ('choice:%s' % ch)]
        batch_input = self.input_to_tensor(x, de, me, qembedding, qnodes_idx, choices_nodes_idx)
        batch_output = self.output_to_tensor(self.key_map[qdata['answerKey']])
        return batch_input, batch_output

    def preprocess(self):
        for qdata in tqdm(self.data):
            qdata['batch_input'], qdata['batch_output'] = self.preprocess_qdata(qdata)

    def get_embedding(self, question):
        tokens = self.nlp.word_tokenize(question)
        emb = self.embedder.ee.embed_batch([tokens])[0]
        return np.concatenate([emb[0], emb[1], emb[2]], axis=1)

    def shuffle(self):
        np.random.shuffle(self.idcs)

    def __iter__(self):
        self.shuffle()
        self._cursor = 0
        return self

    def tensor(self, x):
        return torch.Tensor(x)
    
    def input_to_tensor(self, x, de, me, qembedding, qnodes_idx, choices_nodes_idx):
        x, de, me, qembedding = list(map(self.tensor, [x, de, me, qembedding]))
        return x, [de, me], qembedding, qnodes_idx, choices_nodes_idx
    
    def output_to_tensor(self, answer):
        return self.tensor([answer]).to(torch.long)
    
    def batch_input_to_tensor(self, batch_input, batch_output):
        x, rel_adj, qembedding, qnode_idx, choices_nodes_idx = batch_input
        x = x.to(self.device)
        nrel_adj = []
        for adj in rel_adj:
            nrel_adj.append(adj.to(self.device))
        qembedding = qembedding.to(self.device)
        batch_output = batch_output.to(self.device)
        return [x, nrel_adj, qembedding, qnode_idx, choices_nodes_idx], batch_output

    def __next__(self):
        qdata = self.data[self.idcs[self._cursor]]
        pcursor = self._cursor
        self._cursor = (self._cursor + 1) % len(self.data)
        batch_input, batch_output = self.batch_input_to_tensor(qdata['batch_input'], qdata['batch_output'])
        return batch_input, batch_output, True if self._cursor == 0 else False
    
    def __del__(self):
        self.nlp.close()
