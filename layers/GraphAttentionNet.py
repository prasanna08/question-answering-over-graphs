import torch

from QuestionLayers import QAttention, QuestionLayer, QuestionChoiceLayer
from GraphAttentionLayer import GraphAttentionLayer

class GAT(torch.nn.Module):
    def __init__(self, infeat, nhid, dropout, alpha, nenc_heads, ndec_heads, nrels):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.infeat = infeat
        self.nhid = nhid
        self.nrels = nrels
        self.nenc_heads = nenc_heads
        self.ndec_heads = ndec_heads
        self.qlayer = QuestionLayer(infeat, dropout)
        self.qawarenodes = torch.nn.Conv1d(infeat, infeat, kernel_size=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_dropout = torch.nn.Dropout(dropout)
        self.rattentions = [[GraphAttentionLayer(infeat, nhid, dropout=dropout, alpha=alpha) for _ in range(nrels)]]
        self.rattentions.extend([[GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha) for _ in range(nrels)] for _ in range(self.nenc_heads - 1)])
        self.transpose_attentions =[[GraphAttentionLayer(nhid, nhid, dropout=dropout, alpha=alpha) for _ in range(nrels)] for _ in range(self.ndec_heads - 1)]
        self.transpose_attentions.append([GraphAttentionLayer(nhid, infeat, dropout=dropout, alpha=alpha) for _ in range(nrels)])
        self.gpool = torch.nn.MaxPool1d(kernel_size=self.nrels)
        self.lnorm = torch.nn.LayerNorm(nhid)
        for i, rattention in enumerate(self.rattentions):
            for j, attention in enumerate(rattention):
                self.add_module('attention_{}_{}'.format(i, j), attention)
        for i, trattention in enumerate(self.transpose_attentions):
            for j, attention in enumerate(trattention):
                self.add_module('transposed_attention_{}_{}'.format(i, j), attention)
        self.cattention = QuestionChoiceLayer(infeat, nhid, dropout)
        self.rel_proj = torch.nn.Linear(nrels*nhid, nhid)
        self.tr_hid_rel_proj = torch.nn.Linear(nrels*nhid, nhid)
        self.tr_infeat_rel_proj = torch.nn.Linear(nrels*infeat, infeat)
        self.out_layer = torch.nn.Linear(nhid, 1)
        self.out_softmax = torch.nn.LogSoftmax(dim=0)
        self.key_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, rel_adj, qembedding1, qnode_idx, choices_nodes_idx):
        qembedding = self.qlayer(qembedding1)
        N = x.size()[0]
        qawareprobs = self.qawarenodes(x.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        qawareprobs = torch.sigmoid(torch.mm(qawareprobs, qembedding.unsqueeze(1)))
        qawareembeds = qawareprobs * x  + (1 - qawareprobs) * qembedding.repeat(N, 1)
        #qawareembeds = torch.cat([x, qembedding.repeat(N, 1)], dim=1).to(self.device)
        #qawareembeds = self.qawarenodes(qawareembeds.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        x = self.dropout(qawareembeds)
        for rattn in self.rattentions:
            x_rel = []
            for attn, adj in zip(rattn, rel_adj):
                xr = self.lnorm(attn(x, adj))
                x_rel.append(xr)
            x = torch.stack(x_rel)
            x = self.gpool(x.permute(1, 2, 0)).squeeze(-1)
            #x = self.rel_proj(x)
        x = self.out_dropout(x)
        xd = x
        for i, trattn in enumerate(self.transpose_attentions):
            x_rel = []
            for attn, adj in zip(trattn, rel_adj):
                xr = self.lnorm(attn(xd, adj)) if i < self.ndec_heads - 1 else attn(xd, adj)
                x_rel.append(xr)
            xd = torch.stack(x_rel)
            #xd = self.tr_hid_rel_proj(xd) if i < self.ndec_heads - 1 else self.tr_infeat_rel_proj(xd)
            xd = self.gpool(xd.permute(1, 2, 0)).squeeze(-1)
        qnode_embeds = x[qnode_idx, :]
        cembed = []
        cnodes_len = []
        for clabel in ['A', 'B', 'C', 'D', 'E']:
            if clabel in choices_nodes_idx and len(choices_nodes_idx[clabel]) > 0:
                cnodes = choices_nodes_idx[clabel]
                cnode_embeds = x[cnodes, :]
                cembed.append(self.cattention(qembedding1, cnode_embeds))
                cnodes_len.append(len(cnodes))
            else:
                cembed.append(torch.zeros(self.nhid, requires_grad=False).to(self.device))
                cnodes_len.append(0)
        cembed = torch.stack(cembed)
        cnodes_len = torch.Tensor(cnodes_len)
        cout = self.out_layer(cembed)
        mask_gate = (cnodes_len > 0).to(torch.float32).unsqueeze(dim=1).to(self.device)
        cout = mask_gate * cout + (1 - mask_gate) * -1e15
        cout = self.out_softmax(cout)
        return cout, xd
