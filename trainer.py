from tqdm.autonotebook import tqdm
import numpy as np
import torch
#from apex import amp
import torch.nn.functional as F

from layer.GraphAttentionNet import GAT

device = ('cuda' if torch.cuda.is_available() else 'cpu')

class Network(torch.nn.Module):
    def __init__(self, nfeatures, nhidden, dropout, alpha, reg_alpha, nenc_heads, ndec_heads, nrels, model_name='Network'):
        super(Network, self).__init__()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.gat = GAT(nfeatures, nhidden, dropout, alpha, nenc_heads, ndec_heads, nrels)
        self.loss = torch.nn.NLLLoss()
        self.rec_loss = torch.nn.MSELoss()
        self.reg_alpha = reg_alpha
        self.opt = torch.optim.Adam(self.gat.parameters(), lr=0.001, weight_decay=1e-3)
        self.lr = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.9)
        self.model_name = model_name

    def forward(self, x, rel_adj, qembedding, qnode_idx, choices_nodes_idx):
        predicted = self.gat(x, rel_adj, qembedding, qnode_idx, choices_nodes_idx)
        return predicted
    
    def train_iter(self, in_tup, output):
        x, rel_adj, qembedding, qnode_idx, choices_nodes_idx = in_tup
        #rel_adj = [rel_adj[0] + rel_adj[1]]
        pred, xd = self.forward(x, rel_adj, qembedding, qnode_idx, choices_nodes_idx)
        pred = pred.view(-1).unsqueeze(0)
        sloss = self.loss(pred, output) + self.reg_alpha * self.rec_loss(x, xd)
        return sloss, pred.detach()
    
    def get_validation_acc(self, val_bg):
        self.eval()
        pred_val = []
        true_val = []
        val_bg = val_bg.__iter__()
        for _ in tqdm(range(len(val_bg.data)), desc='Validation Accuracy', leave=False):
            in_tup, output, _ = val_bg.__next__()
            x, rel_adj, qembedding, qnode_idx, choices_nodes_idx = in_tup
            #rel_adj = [rel_adj[0] + rel_adj[1]]
            pred, _ = self.forward(x, rel_adj, qembedding, qnode_idx, choices_nodes_idx)
            true_val.append(output[0].item())
            pred_val.append(pred.detach().argmax().item())
        self.train()
        return np.sum(np.array(true_val) == np.array(pred_val)) / len(val_bg.data)
    
    def trainer(self, batch_gen, val_bg, batch_backward=8, max_epochs=30, step_summary=32, ckp_steps=5000, lr_steps=3000):
        overall_loss = 0.0
        bloss = 0.0
        step = 0
        epoch_bar = tqdm(total = max_epochs, desc='Epochs')
        epoch = 0
        while epoch < max_epochs:
            batch_gen = batch_gen.__iter__()
            true_train = []
            pred_train = []
            for _ in tqdm(range(len(batch_gen.data)), desc='Train Step', leave=False):
                x, y, epoch_pass = batch_gen.__next__()
                sloss, pred = self.train_iter(x, y)
                sloss = sloss / batch_backward
                true_train.append(y[0].item())
                pred_train.append(pred.argmax().item())
                sloss.backward()
                bloss += sloss.detach()
                step += 1
                if step % ckp_steps == 0:
                    self.store_model('%s-%d.pt' % (self.model_name, step))
                if step % batch_backward == 0:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                    self.opt.step()
                    self.opt.zero_grad()
                    overall_loss += bloss
                    bloss = 0.0
                if step % step_summary == 0:
                    tqdm.write('Step %d, Loss %.3f' % (step, overall_loss / (step_summary / batch_backward)))
                    overall_loss =; 0.0
                if step % lr_steps:
                    self.lr.step()
            epoch += 1
            train_acc = np.sum(np.array(true_train) == np.array(pred_train)) / len(bg.data)
            val_acc = self.get_validation_acc(val_bg)
            tqdm.write('Epoch %d, Train Accuracy: %.3f, Validation Accuracy: %.3f' % (epoch, train_acc, val_acc))
            epoch_bar.update(1)
        epoch_bar.close()
    
    def store_model(self, fname, with_opt=True):
        data = {
            'model': self.state_dict(),
        }
        if with_opt:
            data['optimizer'] = self.opt.state_dict()
            data['lr'] = self.lr.state_dict()
        torch.save(data, fname)
