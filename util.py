import numpy as np
import torch


class GraphAug:
    def __init__(self, aug_ratio, aug_methods='edge_perturb'):
        self.name2method = {
            'edge_perturb': self.perturb_edges,
            'node_drop': self.node_drop,
            'subgraph': self.subgraph,
            'attr_mask': self.mask_nodes,
        }

        self.aug_ratio = aug_ratio
        if isinstance(aug_methods, str):
            aug_methods = [aug_methods]
        self.funcs = []
        for k, v in self.name2method.items():
            if k in aug_methods:
                self.funcs.append(v)
        self.num_funcs = len(self.funcs)
        print('aug_methods', aug_methods)

    def __call__(self, data):
        rd = torch.randint(self.num_funcs, (2,))
        data.view1_x, data.view1_edge_index, data.view1_edge_attr = self.funcs[rd[0].item()](data)
        data.view2_x, data.view2_edge_index, data.view2_edge_attr = self.funcs[rd[1].item()](data)
        return data

    def perturb_edges(self, data):
        _, edge_num = data.edge_index.size()
        permute_num = int(edge_num * self.aug_ratio)
        idx_delete = torch.randperm(edge_num)[permute_num:]
        new_edge_index = data.edge_index[:, idx_delete]
        new_edge_attr = data.edge_attr[idx_delete]
        return data.x, new_edge_index, new_edge_attr

    def node_drop(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num  * self.aug_ratio)

        idx_perm = torch.randperm(node_num).numpy()

        idx_drop = idx_perm[:drop_num]
        idx_nondrop = idx_perm[drop_num:]
        idx_nondrop.sort()
        idx_dict = {idx_nondrop[n]:n for n in list(range(idx_nondrop.shape[0]))}

        edge_index = data.edge_index.numpy()
        edge_mask = np.array([n for n in range(edge_num) if not (edge_index[0, n] in idx_drop or edge_index[1, n] in idx_drop)])

        edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
        try:
            new_edge_index = torch.tensor(edge_index).transpose_(0, 1)
            new_x = data.x[idx_nondrop]
            new_edge_attr = data.edge_attr[edge_mask]
            return new_x, new_edge_index, new_edge_attr
        except:
            return data.x, data.edge_index, data.edge_attr

    def subgraph(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * self.aug_ratio)

        edge_index = data.edge_index.numpy()

        idx_sub = [torch.randint(node_num, (1,)).item()]
        idx_neigh = set([n for n in edge_index[1][edge_index[0]==idx_sub[0]]])

        count = 0
        while len(idx_sub) <= sub_num:
            count = count + 1
            if count > node_num:
                break
            if len(idx_neigh) == 0:
                break
            sample_node = np.random.choice(list(idx_neigh))
            if sample_node in idx_sub:
                continue
            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(set([n for n in edge_index[1][edge_index[0]==idx_sub[-1]]]))

        idx_drop = [n for n in range(node_num) if not n in idx_sub]
        idx_nondrop = idx_sub
        idx_dict = {idx_nondrop[n]:n for n in list(range(len(idx_nondrop)))}
        edge_mask = np.array([n for n in range(edge_num) if (edge_index[0, n] in idx_nondrop and edge_index[1, n] in idx_nondrop)])

        edge_index = data.edge_index.numpy()
        edge_index = [[idx_dict[edge_index[0, n]], idx_dict[edge_index[1, n]]] for n in range(edge_num) if (not edge_index[0, n] in idx_drop) and (not edge_index[1, n] in idx_drop)]
        try:
            new_edge_index = torch.tensor(edge_index).transpose_(0, 1)
            new_x = data.x[idx_nondrop]
            new_edge_attr = data.edge_attr[edge_mask]
            return new_x, new_edge_index, new_edge_attr
        except:
            return data.x, data.edge_index, data.edge_attr

    def mask_nodes(self, data):
        node_num, feat_dim = data.x.size()
        mask_num = int(node_num * self.aug_ratio)

        token = data.x.mean(dim=0)
        idx_mask = torch.randperm(node_num)[:mask_num]
        data.x[idx_mask] = token.clone().detach()

        return data.x, data.edge_index, data.edge_attr
