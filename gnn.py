import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_max_pool as gmp, global_add_pool as gap,global_mean_pool as gep,global_sort_pool
from torch_geometric.utils import dropout_adj
from layer import PairNorm

from Naive import NaiveTransformerLayer


# GCN based model
class GNNNet(torch.nn.Module):
    def __init__(self, n_output=1, num_features_pro=54, num_features_mol=78, output_dim=128, dropout=0.2):
        super(GNNNet, self).__init__()

        self.norm = PairNorm("PN-SCS", 100)

        # self.batch_norm = nn.BatchNorm2d()

        print('GNNNet Loaded')
        self.n_output = n_output
        self.mol_conv1 = GCNConv(num_features_mol, num_features_mol)
        self.mol_conv2 = GCNConv(num_features_mol, num_features_mol * 2)
        self.mol_conv3 = GCNConv(num_features_mol * 2, num_features_mol * 4)
        self.mol_fc_g1 = torch.nn.Linear(num_features_mol * 4, 1024)
        self.mol_fc_g2 = torch.nn.Linear(1024, output_dim)

        # self.pro_conv1 = GCNConv(embed_dim, embed_dim)
        self.pro_conv1 = GCNConv(num_features_pro, num_features_pro)
        self.pro_conv2 = GCNConv(num_features_pro, num_features_pro * 2)
        self.pro_conv3 = GCNConv(num_features_pro * 2, num_features_pro * 4)
        # self.pro_conv4 = GCNConv(embed_dim * 4, embed_dim * 8)
        self.pro_fc_g1 = torch.nn.Linear(num_features_pro * 4, 1024)
        self.pro_fc_g2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(114, 114, 3, bidirectional=False)

        # smiles_vec layer
        self.embedding_mol = nn.Embedding(num_features_mol + 1, 128)
        self.conv_mol_1 = nn.Conv1d(in_channels=100, out_channels=32, kernel_size=8)
        self.conv_mol_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.conv_mol_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.conv_mol_4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.fc1_mol = nn.Linear(32*114, output_dim)
        self.mol_bn = nn.BatchNorm1d(32)

        # protein_vec layer
        self.embedding_pro = nn.Embedding(num_features_pro+1, 128)
        self.conv_pro_1 = nn.Conv1d(in_channels=1000, out_channels=32, kernel_size=8)
        self.conv_pro_2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.conv_pro_3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.conv_pro_4 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=8)
        self.fc1_pro = nn.Linear(32*114, output_dim)
        self.pro_bn = nn.BatchNorm1d(32)

        # attention layer
        self.liner1 = nn.Linear(output_dim*4, output_dim)
        self.liner2 = nn.Linear(output_dim, 1)

        # combined layers
        self.fc1 = nn.Linear(4 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)

        # # predefined word embedding

        # transformer layer
        self.transformer = NaiveTransformerLayer(128)

        self.attention = nn.Parameter(torch.ones(4, 1, 1)/4)

    def forward(self, data_mol, data_pro):
        # get graph input
        mol_x, mol_edge_index, mol_batch, mol_vec = data_mol.x, data_mol.edge_index, data_mol.batch, data_mol.smile_vec
        mol_vec = torch.tensor(mol_vec, dtype=torch.long)
        # get protein input
        target_x, target_edge_index, target_batch, target_vec = data_pro.x, data_pro.edge_index, data_pro.batch, data_pro.protein_vec
        target_vec = torch.tensor(target_vec, dtype=torch.long)
        x = self.mol_conv1(mol_x, mol_edge_index)
        x = self.norm(x)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv2(x, mol_edge_index)
        x = self.norm(x)
        x = self.relu(x)

        # mol_edge_index, _ = dropout_adj(mol_edge_index, training=self.training)
        x = self.mol_conv3(x, mol_edge_index)
        x = self.relu(x)
        x = gep(x, mol_batch)  # global pooling

        # flatten
        x = self.relu(self.mol_fc_g1(x))
        x = self.dropout(x)
        x = self.mol_fc_g2(x)
        x = self.dropout(x) # 64 x 128
        # x = self.transformer(x)

        xt = self.pro_conv1(target_x, target_edge_index)
        xt = self.norm(xt)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv2(xt, target_edge_index)
        xt = self.norm(xt)
        xt = self.relu(xt)

        # target_edge_index, _ = dropout_adj(target_edge_index, training=self.training)
        xt = self.pro_conv3(xt, target_edge_index)
        xt = self.relu(xt)

        xt = gep(xt, target_batch)  # global pooling

        # flatten
        xt = self.relu(self.pro_fc_g1(xt))
        xt = self.pro_fc_g2(xt)
        xt = self.dropout(xt)  # 64 x 128

        # smile_vec layer ######################
        embedded_mol = self.embedding_mol(mol_vec)
        conv_mol = self.conv_mol_1(embedded_mol)
        conv_mol = self.dropout2(conv_mol)
        conv_mol = self.conv_mol_2(conv_mol)  # 64 x 32 x 114


        conv_mol = torch.transpose(conv_mol, 0, 1)
        conv_mol, _ = self.rnn(conv_mol)
        conv_mol = torch.transpose(conv_mol, 0, 1)  # 64 x 32 x 114

        mol = conv_mol.contiguous().view(-1, 32*114)
        mol = self.fc1_mol(mol)  # 64 x 128
        # mol = self.transformer(mol)

        # proteins_vec layer ####################
        embedded_pro = self.embedding_pro(target_vec)
        conv_pro = self.conv_pro_1(embedded_pro)
        conv_pro = self.dropout2(conv_pro)
        conv_pro = self.conv_pro_2(conv_pro)  # 64 x 32 x 114

        conv_pro = torch.transpose(conv_pro, 0, 1)
        conv_pro, _ = self.rnn(conv_pro)
        conv_pro = torch.transpose(conv_pro, 0, 1)  # 64 x 32 x 114

        pro = conv_pro.contiguous().view(-1, 32*114)
        pro = self.fc1_pro(pro)  # 64 x 128

        # attention
        attention = torch.softmax(self.attention, dim=0)
        l = [x, xt, mol, pro]
        xc = torch.stack(l, dim=0)
        xc = torch.sum(xc*attention, dim=0)  # 64 x 128


        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
