import torch

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool


class GMNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GMNConv, self).__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.mlp = torch.nn.Sequential(torch.nn.Linear(3*emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

    def get_self_loop_attr(self, x, edge_attr):
        raise NotImplementedError('')

    def encode_edge_feat(self, edge_attr):
        raise NotImplementedError('')

    def forward(self, gid, x, edge_index, edge_attr, batch):
        # x             (node_num, emb_dim)
        # edge_index    (2, edge_num)
        # edge_attr     (edge_num, edge_attr_num)
        # batch         (node_num)

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = self.get_self_loop_attr(x, edge_attr)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.encode_edge_feat(edge_attr)

        inner_aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)     # (node_num, 2 * emb_dim)
        # here `inner_aggr_out` is the representation where inner-graph aggregation has been applied.
        # `inner_aggr_out[:, :emb_dim]` are aggregated node representations,
        # while `inner_aggr_out[:, emb_dim:]` are aggregated edge representations.

        gx = x[batch == gid]

        '''
        single -> multi
        '''

        a = torch.matmul(x, gx.T)                                                       # (node_num, g_node_num)
        a1 = softmax(a, batch)
        # column of `a1` will be softmaxed in each batch.
        # `a1[i, j]` represents the attention weight of message j -> i.

        mu1 = a1.unsqueeze(-1) * gx.unsqueeze(0)                                        # (node_num, g_node_num, emb_dim)
        # `mu1[i, j] = a1[i, j] * x[j]`.

        sum_mu1 = torch.sum(mu1, dim=1)                                                 # (node_num, emb_dim)
        # weighted sum of `\mu1` for each graph in this batch.

        inner_inter_aggr_out1 = torch.cat([inner_aggr_out, sum_mu1], dim=-1)            # (node_num, 3 * emb_dim)
        # concat the aggregated inter-graph message with `inner_aggr_out`.

        out1 = self.mlp(inner_inter_aggr_out1)                                          # (node_num, emb_dim)
        # update function for each graph.

        out_multi = out1

        '''
        multi -> single
        '''

        a2 = torch.softmax(a.transpose(1, 0), dim=0)                                    # (g_node_num, node_num)
        mu2 = a2.unsqueeze(-1) * x.unsqueeze(0)                                         # (g_node_num, node_num, emb_dim)
        sum_mu2 = global_add_pool(mu2.transpose(1, 0), batch)                           # (batch_size, g_node_num, emb_dim)
        inner_aggr_out2 = inner_aggr_out[batch == gid].unsqueeze(0).expand(sum_mu2.shape[0], -1, -1)
                                                                                        # (batch_size, g_node_num, 2 * emb_dim)
        inner_inter_aggr_out2 = torch.cat([inner_aggr_out2, sum_mu2], dim=-1)           # (batch_size, g_node_num, 3 * emb_dim)
        out2 = self.mlp(inner_inter_aggr_out2.view(-1, 3 * self.emb_dim)).view(sum_mu2.shape[0], -1, self.emb_dim)
                                                                                        # (batch_size, g_node_num, emb_dim)

        out_single = out2

        return out_multi, out_single

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim=1)


class BioGMNConv(GMNConv):
    def __init__(self, emb_dim, aggr='add'):
        super(BioGMNConv, self).__init__(emb_dim, aggr=aggr)
        self.edge_encoder = torch.nn.Linear(9, emb_dim)

    def get_self_loop_attr(self, x, edge_attr):
        self_loop_attr = torch.zeros(x.size(0), 9)
        self_loop_attr[:,7] = 1 # attribute for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        return self_loop_attr

    def encode_edge_feat(self, edge_attr):
        return self.edge_encoder(edge_attr)


class ChemGMNConv(GMNConv):
    def __init__(self, emb_dim, aggr='add'):
        super().__init__(emb_dim, aggr=aggr)
        from chem_model import num_bond_type, num_bond_direction
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def get_self_loop_attr(self, x, edge_attr):
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        return self_loop_attr

    def encode_edge_feat(self, edge_attr):
        return self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])


class AdaGMNConv(torch.nn.Module):
    def __init__(self, emb_dim, mode='bio'):
        super(AdaGMNConv, self).__init__()

        self.mode2conv = {
            'bio': BioGMNConv,
            'chem': ChemGMNConv
        }

        self.gmnconv = self.mode2conv[mode](emb_dim)

    def forward(self, gid, x, edge_index, edge_attr, batch):
        return self.gmnconv(gid, x, edge_index, edge_attr, batch)
