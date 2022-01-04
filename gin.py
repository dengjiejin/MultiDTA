import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from layer import PairNorm


class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GINConvNet, self).__init__()

        self.norm = PairNorm("PN", 1)

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output

        # mol_convolution layers
        mol_nn1 = Sequential(Linear(num_features_mol, dim), ReLU(), Linear(dim, dim))
        self.mol_conv1 = GINConv(mol_nn1)
        self.mol_bn1 = torch.nn.BatchNorm1d(dim)

        mol_nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.mol_conv2 = GINConv(mol_nn2)
        self.mol_bn2 = torch.nn.BatchNorm1d(dim)

        mol_nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.mol_conv3 = GINConv(mol_nn3)
        self.mol_bn3 = torch.nn.BatchNorm1d(dim)

        mol_nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.mol_conv4 = GINConv(mol_nn4)
        self.mol_bn4 = torch.nn.BatchNorm1d(dim)

        mol_nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.mol_conv5 = GINConv(mol_nn5)
        self.mol_bn5 = torch.nn.BatchNorm1d(dim)

        self.mol_fc_g1 = torch.nn.Linear(dim, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        # pro_converlution layer
        pro_nn1 = Sequential(Linear(num_features_pro, dim), ReLU(), Linear(dim, dim))
        self.pro_conv1 = GINConv(pro_nn1)
        self.pro_bn1 = torch.nn.BatchNorm1d(dim)

        pro_nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.pro_conv2 = GINConv(pro_nn2)
        self.pro_bn2 = torch.nn.BatchNorm1d(dim)

        pro_nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.pro_conv3 = GINConv(pro_nn3)
        self.pro_bn3 = torch.nn.BatchNorm1d(dim)

        pro_nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.pro_conv4 = GINConv(pro_nn4)
        self.pro_bn4 = torch.nn.BatchNorm1d(dim)

        pro_nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.pro_conv5 = GINConv(pro_nn5)
        self.pro_bn5 = torch.nn.BatchNorm1d(dim)

        self.pro_fc_g1 = torch.nn.Linear(dim, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # get protein input
        target_x, target_edge_index, target_batch = data_pro.x, data_pro.edge_index, data_pro.batch

        mol_x = F.relu(self.mol_conv1(mol_x, mol_edge_index))
        mol_x = self.mol_bn1(mol_x)
        mol_x = F.relu(self.mol_conv2(mol_x, mol_edge_index))
        mol_x = self.mol_bn1(mol_x)
        mol_x = F.relu(self.mol_conv3(mol_x, mol_edge_index))
        mol_x = self.mol_bn1(mol_x)
        mol_x = F.relu(self.mol_conv4(mol_x, mol_edge_index))
        mol_x = self.mol_bn1(mol_x)
        mol_x = F.relu(self.mol_conv5(mol_x, mol_edge_index))
        mol_x = self.mol_bn1(mol_x)
        mol_x = global_add_pool(mol_x, mol_batch)
        # flatten
        mol_x = F.relu(self.mol_fc_g1(mol_x))
        mol_x = F.dropout(mol_x, p=0.2, training=self.training)
        mol_x = self.mol_fc_g2(mol_x)
        mol_x = F.dropout(mol_x, p=0.2, training=self.training)

        target_x = F.relu(self.pro_conv1(target_x, target_edge_index))
        target_x = self.pro_bn1(target_x)
        target_x = F.relu(self.pro_conv2(target_x, target_edge_index))
        target_x = self.pro_bn1(target_x)
        target_x = F.relu(self.pro_conv3(target_x, target_edge_index))
        target_x = self.pro_bn1(target_x)
        target_x = F.relu(self.pro_conv4(target_x, target_edge_index))
        target_x = self.pro_bn1(target_x)
        target_x = F.relu(self.pro_conv5(target_x, target_edge_index))
        target_x = self.pro_bn1(target_x)
        target_x = global_add_pool(target_x, target_batch)
        # flatten
        target_x = F.relu(self.pro_fc_g1(target_x))
        target_x = F.dropout(target_x, p=0.2, training=self.training)
        target_x = self.pro_fc_g2(target_x)
        target_x = F.dropout(target_x, p=0.2, training=self.training)

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


