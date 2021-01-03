from torch.utils.data import DataLoader
import dgl
import torch
import numpy as np
import sys
import os
import socnavData
import pickle
import torch.nn.functional as F

def activation_functions(activation_tuple_src):
    ret = []
    for x in activation_tuple_src:
        if x == 'relu':
            ret.append(F.relu)
        elif x == 'elu':
            ret.append(F.elu)
        elif x == 'tanh':
            ret.append(torch.tanh)
        elif x == 'leaky_relu':
            ret.append(F.leaky_relu)
        else:
            print('Unknown activation function {}.'.format(x))
            sys.exit(-1)
    return tuple(ret)

sys.path.append(os.path.join(os.path.dirname(__file__), 'nets'))

sys.path.append(os.path.join(os.path.dirname(__file__), 'nets'))
from select_gnn import SELECT_GNN
from rgcnDGL import RGCN

global g_device 

def collate(batch):
    graphs = [batch[0][0]]
    labels = batch[0][1]
    for graph, label in batch[1:]:
        graphs.append(graph)
        labels = torch.cat([labels, label], dim=0)
    batched_graphs = dgl.batch(graphs).to(torch.device(g_device))
    labels.to(torch.device(g_device))

    return batched_graphs, labels

class SocNavAPI(object):
    def __init__(self, base, dataset, device='cpu'):
        self.device = torch.device(device)  # For gpu change it to cuda
        global g_device
        g_device = self.device
        self.device2 = torch.device('cpu')
        print(base)
        self.params = pickle.load(open(os.path.dirname(__file__)+'/SOCNAV.prms', 'rb'), fix_imports=True)
        self.params['net'] = self.params['net'].lower()
        print(self.params)
        print(self.params['net'])
        self.GNNmodel = SELECT_GNN(num_features = self.params['num_feats'],
                     n_classes = self.params['n_classes'],            #n_classes
                     num_hidden = self.params['num_hidden'],
                     gnn_layers = self.params['gnn_layers'],
                     dropout = self.params['in_drop'],
                     activation = self.params['nonlinearity'],
                     num_channels = 1,
                     gnn_type = self.params['net'],
                     K = 10,                # sage filters
                     num_heads = self.params['heads'],
                     num_rels = self.params['num_rels'],
                     num_bases = self.params['num_rels'],
                     g = None,
                     residual = self.params['residual'],
                     attn_drop = self.params['attn_drop'],
                     num_hidden_layers_rgcn = self.params['n_hlayers_rgcn'],
                     num_hidden_layers_gat = self.params['n_hlayers_gat'],
                     num_hidden_layer_pairs = self.params['n_hlayers_pairs']
                     )
        # if self.params['net'] in ['rgcn']:
        #     self.GNNmodel = RGCN(g=None,
        #                          num_gnn_layers=self.params['gnn_layers'],
        #                          in_dim=self.params['num_feats'],
        #                          hidden_dimensions=self.params['num_hidden'],  # feats hidden
        #                          num_rels=self.params['num_rels'],
        #                          activation=torch.tanh,
        #                          feat_drop=self.params['in_drop'],
        #                          num_bases=self.params['num_rels'])
        # else:
        #     print('Unknown/unsupported model in the parameters file')
        #     sys.exit(0)


        self.GNNmodel.load_state_dict(torch.load(os.path.dirname(__file__)+'/SOCNAV.tch', map_location = device))
        self.GNNmodel.to(self.device)
        self.GNNmodel.eval()

        for name, param in self.GNNmodel.named_parameters():
            print(name, param)


        if dataset is not None:
            self.test_dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)
        

    def predictOneGraph(self, g):
        self.test_dataloader = DataLoader(g, batch_size=1, collate_fn=collate)
        logits = self.predict()
        return logits
        
    def predict(self):
        result = []
        for batch, data in enumerate(self.test_dataloader):
            subgraph, labels = data
            feats = subgraph.ndata['h'].to(self.device)

            if self.params['fw'] == 'dgl':
                self.GNNmodel.gnn_object.g = subgraph
                self.GNNmodel.g = subgraph
                for layer in self.GNNmodel.gnn_object.layers:
                    layer.g = subgraph
                if self.params['net'] in ['rgcn']:
                    logits = self.GNNmodel(feats.float(), subgraph.edata['rel_type'].squeeze().to(self.device))
                else:
                    logits = self.GNNmodel(feats.float(), None)
            else:
                self.GNNmodel.g = subgraph
                if self.params['net'] in [ 'pgat', 'pgcn', 'ptag', 'psage', 'pcheb' ]:
                    dataI = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(self.device))
                else:
                    dataI = Data(x=feats.float(), edge_index=torch.stack(subgraph.edges()).to(self.device), edge_type=subgraph.edata['rel_type'].squeeze().to(self.device))

                logits = self.GNNmodel(dataI, subgraph)

            result.append(logits[0])
        return result
        # result = []
        # for batch, data in enumerate(self.test_dataloader):
        #     subgraph, labels = data
        #     feats = subgraph.ndata['h'].to(self.device)
        #     self.GNNmodel.set_g(subgraph)
        #     if self.params['net'] in ['rgcn']:
        #         logits = self.GNNmodel(feats.float(), subgraph.edata['rel_type'].squeeze().to(self.device))
        #     else:
        #         logits = self.GNNmodel(feats.float())
        #     result.append(logits[0])
        # return result
        
