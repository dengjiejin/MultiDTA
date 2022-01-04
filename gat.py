import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_max_pool as gmp
from layer import PairNorm


# GAT model
class GATNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()

        self.norm = PairNorm("PN", 1)

        # mol layer
        self.mol_gcn1 = GATConv(num_features_mol, num_features_mol, heads=2, dropout=dropout)
        self.mol_gcn2 = GATConv(num_features_mol*2, num_features_mol * 2, dropout=dropout)
        self.mol_fc_g1 = nn.Linear(num_features_mol * 2, 1024)
        self.mol_fc_g2 = nn.Linear(1024, output_dim)

        # pro_layer
        self.pro_gcn1 = GATConv(num_features_pro, num_features_pro, heads=2, dropout=dropout)
        self.pro_gcn2 = GATConv(num_features_pro*2, num_features_pro * 2, dropout=dropout)
        self.pro_fc_g1 = nn.Linear(num_features_pro * 2, 1024)
        self.pro_fc_g2 = nn.Linear(1024, output_dim)

        # combined
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activate and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        mol_x = F.dropout(mol_x, p=0.2, training=self.training)
        mol_x = F.relu(self.mol_gcn1(mol_x, mol_edge_index))
        mol_x = F.dropout(mol_x, p=0.2, training=self.training)
        mol_x = F.relu(self.mol_gcn2(mol_x, mol_edge_index))
        mol_x = gmp(mol_x, mol_batch)
        mol_x = self.relu(self.mol_fc_g1(mol_x))
        mol_x = self.dropout(mol_x)
        mol_x = self.mol_fc_g2(mol_x)
        mol_x = self.dropout(mol_x)

        target_x = F.dropout(target_x, p=0.2, training=self.training)
        target_x = F.relu(self.pro_gcn1(target_x, target_edge_index))
        target_x = F.dropout(target_x, p=0.2, training=self.training)
        target_x = F.relu(self.pro_gcn2(target_x, target_edge_index))
        target_x = gmp(target_x, target_batch)
        target_x = self.relu(self.pro_fc_g1(target_x))
        target_x = self.dropout(target_x)
        target_x = self.pro_fc_g2(target_x)
        target_x = self.dropout(target_x)

        # concat
        xc = torch.cat((mol_x, target_x), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
