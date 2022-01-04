import argparse
import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from GCNN import GCNIIdenseConv
import torch.nn as nn


class GCNIIdense_model(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, hidden_channels=512, output_dim=128,
                 dropout=0.2, num_layers=3, alpha=0.2, norm='bn'):
        super(GCNIIdense_model, self).__init__()

        self.n_output = n_output
        self.mol_convs = torch.nn.ModuleList()
        self.mol_convs.append(torch.nn.Linear(num_features_mol, hidden_channels))
        for _ in range(num_layers):
            self.mol_convs.append(GCNIIdenseConv(hidden_channels, hidden_channels, bias=norm))
        self.mol_convs.append(torch.nn.Linear(hidden_channels, output_dim))
        self.mol_reg_params = list(self.mol_convs[1:-1].parameters())
        self.mol_non_reg_params = list(self.mol_convs[0:1].parameters()) + list(self.mol_convs[-1:].parameters())

        self.pro_convs = torch.nn.ModuleList()
        self.pro_convs.append(torch.nn.Linear(num_features_pro, hidden_channels))
        for _ in range(num_layers):
            self.pro_convs.append(GCNIIdenseConv(hidden_channels, hidden_channels, bias=norm))
        self.pro_convs.append(torch.nn.Linear(hidden_channels, output_dim))
        self.pro_reg_params = list(self.pro_convs[1:-1].parameters())
        self.pro_non_reg_params = list(self.pro_convs[0:1].parameters()) + list(self.pro_convs[-1:].parameters())

        self.dropout = dropout
        self.alpha = alpha

        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        mol_hidden = []
        mol_x = F.dropout(mol_x, self.dropout, training=self.training)
        mol_x = F.relu(self.mol_convs[0](mol_x))
        mol_hidden.append(mol_x)
        for i, con in enumerate(self.mol_convs[1:-1]):
            mol_x = F.dropout(mol_x, self.dropout, training=self.training)
            mol_x = F.relu(con(mol_x, mol_edge_index, self.alpha, mol_hidden[0])) + mol_hidden[-1]
            mol_hidden.append(mol_x)
        mol_x = F.dropout(mol_x, self.dropout, training=self.training)
        mol_x = self.mol_convs[-1](mol_x)

        target_hidden = []
        target_x = F.dropout(target_x, self.dropout, training=self.training)
        target_x = F.relu(self.pro_convs[0](target_x))
        target_hidden.append(target_x)
        for i, con in enumerate(self.pro_convs[1:-1]):
            target_x = F.dropout(target_x, self.dropout, training=self.training)
            target_x = F.relu(con(target_x, target_edge_index, self.alpha, target_hidden[0])) + target_hidden[-1]
            target_hidden.append(target_x)
        target_x = F.dropout(target_x, self.dropout, training=self.training)
        target_x = self.pro_convs[-1](target_x)

        # concat
        xc = torch.cat((mol_x, target_x), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
