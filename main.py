import argparse

import BatchGenerator as bg
from trainer import Network
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--input-dim', help='Dimension of initial embedding', default=768, type=int)
parser.add_argument('--hidden-dim', help='Dimension of hidden layer', 256, type=int)
parser.add_argument('--dropout', help='Dropout rate of layers', default=0.4, type=float)
parser.add_argument('--rec-loss', help='Weight of reconstruction loss for encoder decoder regularization', default=1e-5, type=float)
parser.add_argument('--enc-layers', help='Number of encoder layers', default=3, type=int)
parser.add_argument('--dec-layers', help='Number of decoder layers', default=2, type=int)
parser.add_argument('--leakyrelu', help='The leak value of leakyrelu activation function', default=0.2, type=float)
parser.add_argument('--max-epochs', help='Total number of epochs', default=10, type=int)
parser.add_argument('--summary-steps', help='Number of summary printing steps', default=32, type=float)
parser.add_argument('--batch-size', help='Batch size of mini batch sgd using Adam optimizer', default=0.2, type=float)
parser.add_argument('--checkpoint-steps', help='Number of steps between checkpoints', default=2000, type=int)
parser.add_argument('--output-model', help='File name to store output model', default='OpenBookQA', type=str)
parser.add_argument('--lr-sched-steps', help='Number of steps between applying learning rate scheduler', default=3000, type=int)
parser.add_argument('--n-relations', help='Number of relations in input graph', default=2, type=int)


if __name__ == '__main__':
    args = parser.parse_args()
    data = utils.load_final_data()
    dev_data = utils.load_dev_data()
    bg = bg.BatchGenerator(data, False)
    dev_bg = bg.BatchGenerator(dev_data, False)

    print('Training network ...')

    input_dim = args.input_dim
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    rec_loss = args.rec_loss
    enc_layers = args.enc_layers
    dec_layers = args.dec_layers
    n_relations = args.n_relations
    leakyrelu = args.leakyrelu
    max_epochs = args.max_epochs
    output_model = args.output_model
    summary_steps = args.summary_steps
    checkpoint_steps = args.checkpoint_steps
    batch_size = args.batch_size

    net = Network(input_dim, hidden_dim, , dropout, leakyrelu, rec_loss, enc_layer, dec_layer, n_relations, output_model).to(device)
    net.train()
    net.trainer(bg, dev_bg, batch_backward=batch_size, max_epochs=max_epochs, step_summary=summary_steps, ckp_steps=checkpoint_steps, lr_epochs=lr_sched_steps)
