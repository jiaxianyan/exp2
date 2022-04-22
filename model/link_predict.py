import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, config):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(config.model.in_feats, config.model.h_feats, 'mean')
        self.conv2 = SAGEConv(config.model.h_feats, config.model.h_feats, 'mean')

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

class MLPPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W1 = nn.Linear(config.model.h_feats * 2, config.model.h_feats)
        self.W2 = nn.Linear(config.model.h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(F.relu(self.W1(h))).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']