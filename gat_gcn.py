import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from layer import PairNorm


# GCN-CNN based model
class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GAT_GCN, self).__init__()

        self.norm = PairNorm("PN", 1)

        self.n_output = n_output
        self.mol_conv1 = GATConv(num_features_mol, num_features_mol, heads=2)
        self.mol_conv2 = GCNConv(num_features_mol*2, num_features_mol * 2)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.pro_conv1 = GATConv(num_features_pro, num_features_pro, heads=2)
        self.pro_conv2 = GCNConv(num_features_pro*2, num_features_pro * 2)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 2, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        mol_x = self.mol_conv1(mol_x, mol_edge_index)
        mol_x = self.relu(mol_x)
        mol_x = self.mol_conv2(mol_x, mol_edge_index)
        mol_x = self.relu(mol_x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        mol_x = torch.cat([gmp(mol_x, mol_batch), gap(mol_x, mol_batch)], dim=1)
        mol_x = self.relu(self.mol_fc_g1(mol_x))
        mol_x = self.dropout(mol_x)
        mol_x = self.mol_fc_g2(mol_x)
        mol_x = self.dropout(mol_x)

        target_x = self.pro_conv1(target_x, target_edge_index)
        target_x = self.relu(target_x)
        target_x = self.pro_conv2(target_x, target_edge_index)
        target_x = self.relu(target_x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        target_x = torch.cat([gmp(target_x, target_batch), gap(target_x, target_batch)], dim=1)
        target_x = self.relu(self.pro_fc_g1(target_x))
        target_x = self.dropout(target_x)
        target_x = self.pro_fc_g2(target_x)
        target_x = self.dropout(target_x)

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
