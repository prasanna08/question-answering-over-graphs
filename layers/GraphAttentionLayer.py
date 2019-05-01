import torch

class GraphAttentionLayer(torch.nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.seq_transformation = torch.nn.Conv1d(in_features, out_features, kernel_size=1)
        self.f_1 = torch.nn.Conv1d(out_features, 1, kernel_size=1)
        self.f_2 = torch.nn.Conv1d(out_features, 1, kernel_size=1) 
        self.leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.output_leakyrelu = torch.nn.LeakyReLU(self.alpha)
        self.node_softmax = torch.nn.Softmax(dim=1)
        self.seq_droput = torch.nn.Dropout(dropout)
        self.update_gate = torch.nn.Conv1d(2 * out_features, out_features, kernel_size=1)
        if self.in_features != self.out_features:
            self.linear_proj = torch.nn.Linear(in_features, out_features)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inp, adj):
        seq_fts = self.seq_transformation(inp.transpose(0, 1).unsqueeze(dim=0))
        f_1 = self.f_1(seq_fts)
        f_2 = self.f_2(seq_fts)
        logits = (f_1.transpose(1, 2) + f_2).squeeze(0)
        coef = self.node_softmax(self.leakyrelu(logits) + adj)
        seq_fts = self.seq_droput(seq_fts.squeeze(0).transpose(0, 1))
        seqfits = self.output_leakyrelu(seq_fts)
        ret = torch.mm(coef, seq_fts)
        if self.in_features != self.out_features:
            inp = self.linear_proj(inp)
        up = self.update_gate(torch.cat([ret, inp], dim=1).transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        up = self.sigmoid(up)
        ret = up * ret + (1 - up) * inp
        return ret

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
