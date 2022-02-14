import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from util import GraphAug
from dataloader import DataLoaderContrastive
from graph_matching import AdaGMNConv


class GMPT_CL(nn.Module):

    def __init__(self, args, gnn):
        super(GMPT_CL, self).__init__()
        self.gnn = gnn
        self.cgmn = AdaGMNConv(args.emb_dim, mode=args.mode)

        self.mode = args.mode
        self.temperature = args.temperature
        self.criterion = torch.nn.CrossEntropyLoss()

        labels = torch.cat([torch.arange(args.batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        self.mask = mask.to(args.device)
        self.labels = labels.to(args.device)
        self.zeros = torch.zeros(1, dtype=torch.long).to(args.device)

    def from_pretrained(self, model_file, input_epoch):
        print(f'loading pre-trained model from {model_file}, input_epoch = {input_epoch}.', flush=True)
        self.gnn.load_state_dict(torch.load(model_file + f'.pth.{input_epoch}', map_location=lambda storage, loc: storage))
        self.cgmn.load_state_dict(torch.load(model_file + f'.cgmn.pth.{input_epoch}', map_location=lambda storage, loc: storage))

    def forward_cl(self, gid, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)                                                  # (node_num, emb_dim)
        out_multi, out_single = self.cgmn(gid, x, edge_index, edge_attr, batch)

        out1 = global_mean_pool(out_multi, batch)                                            # (batch_size, emb_dim)
        # mean pooling for a graph level representation.
        out1 = F.normalize(out1, dim=-1)

        out2 = torch.mean(out_single, dim=1)                                                  # (batch_size, emb_dim)
        out2 = F.normalize(out2, dim=-1)

        similarity_matrix = torch.sum(out1 * out2, dim=-1)

        similarity_matrix = similarity_matrix[~self.mask[gid]]

        positives = similarity_matrix[self.labels[gid].bool()]
        negatives = similarity_matrix[~self.labels[gid].bool()]
        logits = torch.cat([positives, negatives])

        logits = logits / self.temperature          # logits[0] is positive.
        loss = self.criterion(logits.unsqueeze(0), self.zeros)
        return loss


def train(args, model, loader, dataset, optimizer, device):
    model.train()

    train_loss_accum = 0
    num_graph = args.batch_size * 2

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        g_ids = np.random.permutation(num_graph)[:args.sample_num]

        optimizer.zero_grad()
        for g_i in g_ids:
            loss = model.forward_cl(g_i, batch.x, batch.edge_index, batch.edge_attr, batch.batch) / args.sample_num
            loss.backward()

            train_loss_accum += float(loss.detach().cpu().item())
        optimizer.step()
        # acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))

    return train_loss_accum/(step+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--mode', type=str, default='bio')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--model_file', type=str, default='', help='filename to output the pre-trained model')
    parser.add_argument('--dataset_path', type=str, default='./', help='root path to the dataset')
    parser.add_argument('--seed', type=int, default=0, help="Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataset loading')
    parser.add_argument('--aug_ratio', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=0.1, help='softmax temperature (default: 0.1)')
    parser.add_argument('--input_epochs', type=int, default=0, help='number of epoch of input model')
    parser.add_argument('--sample_num', type=int, default=4, help='number of sample graph in one batch')
    args = parser.parse_args()
    print(args)


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device

    #set up dataset
    if args.mode == 'bio':
        from bio_loader import BioDataset
        from bio_model import GNN

        root_unsupervised = os.path.join(args.dataset_path, 'dataset/unsupervised')
        dataset = BioDataset(root_unsupervised, data_type='unsupervised', transform=GraphAug(aug_ratio=args.aug_ratio))
        print(dataset[0])
    elif args.mode == 'chem':
        from chem_loader import MoleculeDataset
        from chem_model import GNN

        root = os.path.join(args.dataset_path, 'dataset/zinc_standard_agent')
        dataset = MoleculeDataset(root, dataset='zinc_standard_agent', transform=GraphAug(aug_ratio=args.aug_ratio, aug_methods=['node_drop', 'subgraph']))
    else:
        raise NotImplementedError(f'mode [{args.mode}] not exist.')

    # drop last for calculating contrastive loss easily
    loader = DataLoaderContrastive(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    model = GMPT_CL(args, gnn)
    if args.input_epochs > 0:
        model.from_pretrained(args.model_file, args.input_epochs)
    model = model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    for epoch in range(args.input_epochs + 1, args.input_epochs + args.epochs + 1):
        print("====epoch " + str(epoch), flush=True)
    
        train_loss = train(args, model, loader, dataset, optimizer, device)

        print(train_loss)

        if epoch < 5 or epoch % 5 == 0:
            torch.save(model.gnn.state_dict(), args.model_file + f'.pth.{epoch}')
            torch.save(model.cgmn.state_dict(), args.model_file + f'.cgmn.pth.{epoch}')


if __name__ == "__main__":
    main()
