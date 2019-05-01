import torch
import torch.nn.functional as F

class QAttention(torch.nn.Module):
    def __init__(self, features, dropout):
        super(QAttention, self).__init__()
        self.features = features
        self.W = torch.nn.Conv1d(features, features, kernel_size=1)
        self.temperature = np.power(features, 0.5)
        self.dropout = torch.nn.Dropout(dropout)
        self.softmax = torch.nn.Softmax(dim=0)
    
    def forward(self, qembeddings, kembeddings, vembeddings):
        qembeddings = qembeddings.transpose(0, 1).unsqueeze(dim=0)
        kembeddings = kembeddings.transpose(0, 1).unsqueeze(dim=0)
        vembeddings = vembeddings.transpose(0, 1).unsqueeze(dim=0)
        # Apply (Q, K, V) attention.
        attq = self.W(qembeddings).squeeze(0).transpose(0, 1)
        attk = self.W(kembeddings).squeeze(0)
        attv = self.W(vembeddings).squeeze(0).transpose(0, 1)
        attq = self.dropout(attq)
        attn = torch.matmul(attq, attk)
        attn = attn / self.temperature
        attn = self.softmax(attn)
        outputs = torch.matmul(attn, attv)
        return outputs

class QuestionLayer(torch.nn.Module):
    def __init__(self, in_features, dropout):
        super(QuestionLayer, self).__init__()
        self.in_features = in_features
        self.Qattn = QAttention(in_features, dropout)
        self.Qe = torch.nn.Conv1d(in_features, 1, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=0)
    
    def forward(self, qembeddings):
        outputs = self.Qattn(qembeddings, qembeddings, qembeddings)
        outputs = F.max_pool1d(outputs.transpose(0, 1).unsqueeze(0), kernel_size=outputs.size()[0]).squeeze()
        return outputs

class QuestionChoiceLayer(torch.nn.Module):
    def __init__(self, in_features, n_hidden, dropout):
        super(QuestionChoiceLayer, self).__init__()
        self.in_features = in_features
        self.Qattn = QAttention(n_hidden, dropout)
        self.Qproj = torch.nn.Conv1d(in_features, n_hidden, kernel_size=1)
        self.Qe = torch.nn.Conv1d(n_hidden, 1, kernel_size=1)
        self.softmax = torch.nn.Softmax(dim=0)
    
    def forward(self, qembeddings, cembeddings):
        qembeddings = self.Qproj(qembeddings.transpose(0, 1).unsqueeze(0)).squeeze(0).transpose(0, 1)
        outputs = self.Qattn(qembeddings, cembeddings, cembeddings)
        qattn = self.softmax(self.Qe(outputs.transpose(0, 1).unsqueeze(0)).squeeze(0))
        outputs = torch.matmul(qattn, outputs).squeeze()
        return outputs