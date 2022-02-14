import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm

from splitters import random_split, species_split
from dataloader import DataLoaderAE
from graph_matching import AdaGMNConv


class GMPT_Suppp(nn.Module):
    def __init__(self, args, num_tasks, gnn):
        super(GMPT_Suppp, self).__init__()

        self.mode = args.mode
        self.gnn = gnn
        self.sgmn = AdaGMNConv(args.emb_dim, args.mode)
        self.num_tasks = num_tasks
        self.graph_pred_linear = torch.nn.Linear(args.emb_dim, num_tasks)

        if self.mode == 'bio':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.mode == 'chem':
            self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.mask = ~torch.eye(args.batch_size, dtype=torch.bool)

    def from_pretrained(self, model_file):
        print(f'loading pre-trained model from {model_file}')
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    def calcu_loss(self, gid, batch_graph):
        x, edge_index, edge_attr, batch = batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr, batch_graph.batch
        x = self.gnn(x, edge_index, edge_attr)
        out_multi, out_single = self.sgmn(gid, x, edge_index, edge_attr, batch)

        g_pool1 = global_mean_pool(out_multi, batch)   # (batch_size, emb_dim)
        g_pool2 = torch.mean(out_single, dim=1)        # (batch_size, emb_dim)

        pred1 = self.graph_pred_linear(g_pool1)[self.mask[gid]].double()
        pred2 = self.graph_pred_linear(g_pool2)[self.mask[gid]].double()
        pred = torch.cat([pred1, pred2], dim=-1)

        if self.mode == 'bio':
            batch_y = batch_graph.go_target_pretrain.view(pred1.shape)
        elif self.mode == 'chem':
            batch_y = batch_graph.y.view(-1, self.num_tasks).float()
            is_valid = (batch_y**2 > 0)
            g_is_valid = is_valid[gid]
            batch_y = (batch_y+1)/2

        g_y = batch_y[gid]
        batch_y = batch_y[self.mask[gid]]
        y = torch.cat([batch_y, g_y.expand_as(batch_y)], dim=-1).double()

        if self.mode == 'bio':
            loss = self.criterion(pred, y)
        elif self.mode == 'chem':
            is_valid = is_valid[self.mask[gid]]
            g_is_valid = g_is_valid.expand_as(batch_y)
            is_valid = torch.cat([is_valid, g_is_valid], dim=-1)
            loss_mat = self.criterion(pred, y)
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape, device=loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat)/torch.sum(is_valid)

        return loss


def train(args, model, device, loader, optimizer):
    model.train()

    loss_accum = 0
    num_graph = args.batch_size

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        g_ids = np.random.permutation(num_graph)[:args.sample_num]
        for g_i in g_ids:
            loss = model.calcu_loss(g_i, batch) / args.sample_num
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_accum += float(loss.detach().cpu().item())

    return loss_accum / (step + 1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--mode', type=str, default='bio')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.2,
                        help='dropout ratio (default: 0.2)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--input_model_file', type=str, default = '', help='filename to read the model (if there is any)')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--dataset_path', type=str, default='./', help='root path to the dataset')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_workers', type=int, default = 0, help='number of workers for dataset loading')
    parser.add_argument('--sample_num', type=int, default=1, help='number of sample graph in one batch')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting dataset.")
    parser.add_argument('--split', type=str, default = "species", help='Random or species split')
    parser.add_argument('--comb', type=str, default='cat', help='combination method, one of [concat, dot, fusion, bilinear]')
    args = parser.parse_args()
    print(args)

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device

    if args.mode == 'bio':
        def combine_dataset(AdaDataset, dataset1, dataset2, root_supervised):
            data_list = [data for data in dataset1]
            data_list.extend([data for data in dataset2])
            dataset = AdaDataset(root_supervised, data_type='supervised', empty = True)

            dataset.data, dataset.slices = dataset.collate(data_list)
            return dataset
        
        from bio_loader import BioDataset
        from bio_model import GNN

        root_supervised = os.path.join(args.dataset_path, 'dataset/supervised')
        dataset = BioDataset(root_supervised, data_type='supervised')

        assert args.split == "species"
        print("species splitting")
        trainval_dataset, test_dataset = species_split(dataset)
        test_dataset_broad, test_dataset_none, _ = random_split(test_dataset, seed = args.seed, frac_train=0.5, frac_valid=0.5, frac_test=0)
        print(trainval_dataset)
        print(test_dataset_broad)
        pretrain_dataset = combine_dataset(BioDataset, trainval_dataset, test_dataset_broad, root_supervised)
        print(pretrain_dataset)

        num_tasks = len(pretrain_dataset[0].go_target_pretrain)
    elif args.mode == 'chem':
        from chem_loader import MoleculeDataset
        from chem_model import GNN, GNN_graphpred

        pretrain_dataset = MoleculeDataset(os.path.join(args.dataset_path, "dataset/chembl_filtered"), dataset='chembl_filtered')
        num_tasks = 1310

    train_loader = DataLoaderAE(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers, drop_last=True)

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK=args.JK, drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)
    model = GMPT_Suppp(args, num_tasks, gnn)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)

    model = model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)   
    print(optimizer)

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))

        train_loss = train(args, model, device, train_loader, optimizer)
        print(train_loss)

        if epoch == 1 or epoch % 20 == 0:
            torch.save(model.gnn.state_dict(), args.output_model_file + f'.pth.{epoch}')


if __name__ == "__main__":
    main()
