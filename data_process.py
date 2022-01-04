import pandas as pd
import numpy as np
import os
import random
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
# from keras.preprocessing import text, sequence
from utils import *


def w2v_pad(protein, maxlen_, victor_size):

    #keras API
    tokenizer = text.Tokenizer(num_words=10000, lower=False, filters="　")
    tokenizer.fit_on_texts(protein)
    protein_ = sequence.pad_sequences(tokenizer.texts_to_sequences(protein), maxlen=maxlen_)

    word_index = tokenizer.word_index
    nb_words = len(word_index)
    print(nb_words)
    protVec_model = {}
    with open("dataset/embed/protVec_100d_3grams.csv", encoding='utf8') as f:
        for line in f:
            values = eval(line).rsplit('\t')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            protVec_model[word] = coefs
    print("add protVec finished....")


    count=0
    embedding_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector = protVec_model[word] if word in protVec_model else None
        if embedding_glove_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    del protVec_model
    print(embedding_matrix.shape)
    return protein_, word_index, embedding_matrix


def smile_w2v_pad(smile, maxlen_, victor_size):

    #keras API
    tokenizer = text.Tokenizer(num_words=100, lower=False, filters="　")
    tokenizer.fit_on_texts(smile)
    smile_ = sequence.pad_sequences(tokenizer.texts_to_sequences(smile), maxlen=maxlen_)

    word_index = tokenizer.word_index
    nb_words = len(word_index)
    print(nb_words)
    smileVec_model = {}
    with open("dataset/embed/Atom.vec", encoding='utf8') as f:
        for line in f:
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            smileVec_model[word] = coefs
    print("add smileVec finished....")


    count=0
    embedding_matrix = np.zeros((nb_words + 1, victor_size))
    for word, i in word_index.items():
        embedding_glove_vector = smileVec_model[word] if word in smileVec_model else None
        if embedding_glove_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    del smileVec_model
    print(embedding_matrix.shape)
    return smile_, word_index, embedding_matrix


# nomarlize
def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']

res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


# protein2vector
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i+1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 1000

def seq_cat(prot):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x
#################################################
#smile2vector
smi_dict = smiles_dict = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
smi_dict_len = len(smi_dict)
max_smi_len = 100

def smi_cat(smi):
    x = np.zeros(max_smi_len)
    for i, ch in enumerate(smi[:max_smi_len]):
        x[i] = smi_dict[ch]
    return x
#######################################################
# print(res_weight_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    # print(np.array(res_property1 + res_property2).shape)
    return np.array(res_property1 + res_property2)


# mol atom feature for mol graph
def atom_features(atom):
    # 44 +11 +11 +11 +1
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'X']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


# one ont encoding
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        # print(x)
        raise Exception('input {0} not in allowable set{1}:'.format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    '''Maps inputs not in the allowable set to the last element.'''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# mol smile to mol graph edge index
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    mol_adj = np.zeros((c_size, c_size))
    for e1, e2 in g.edges:
        mol_adj[e1, e2] = 1
        # edge_index.append([e1, e2])
    mol_adj += np.matrix(np.eye(mol_adj.shape[0]))
    index_row, index_col = np.where(mol_adj >= 0.5)
    for i, j in zip(index_row, index_col):
        edge_index.append([i, j])
    # print('smile_to_graph')
    # print(np.array(features).shape)
    return c_size, features, edge_index


# target feature for target graph
def PSSM_calculation(aln_file, pro_seq):
    pfm_mat = np.zeros((len(pro_res_table), len(pro_seq)))
    with open(aln_file, 'r') as f:
        line_count = len(f.readlines())
        for line in f.readlines():
            if len(line) != len(pro_seq):
                print('error', len(line), len(pro_seq))
                continue
            count = 0
            for res in line:
                if res not in pro_res_table:
                    count += 1
                    continue
                pfm_mat[pro_res_table.index(res), count] += 1
                count += 1
    # ppm_mat = pfm_mat / float(line_count)
    pseudocount = 0.8
    ppm_mat = (pfm_mat + pseudocount / 4) / (float(line_count) + pseudocount)
    pssm_mat = ppm_mat
    # k = float(len(pro_res_table))
    # pwm_mat = np.log2(ppm_mat / (1.0 / k))
    # pssm_mat = pwm_mat
    # print(pssm_mat)
    return pssm_mat


def seq_feature(pro_seq):
    pro_hot = np.zeros((len(pro_seq), len(pro_res_table)))
    pro_property = np.zeros((len(pro_seq), 12))
    for i in range(len(pro_seq)):
        # if 'X' in pro_seq:
        #     print(pro_seq)
        pro_hot[i, ] = one_of_k_encoding(pro_seq[i], pro_res_table)
        pro_property[i, ] = residue_features(pro_seq[i])
    return np.concatenate((pro_hot, pro_property), axis=1)


def target_feature(aln_file, pro_seq):
    pssm = PSSM_calculation(aln_file, pro_seq)
    other_feature = seq_feature(pro_seq)
    # print('target_feature')
    # print(pssm.shape)
    # print(other_feature.shape)

    # print(other_feature.shape)
    # return other_feature
    return np.concatenate((np.transpose(pssm, (1, 0)), other_feature), axis=1)


# target aln file save in data/dataset/aln
def target_to_feature(target_key, target_sequence, aln_dir):
    # aln_dir = 'data/' + dataset + '/aln'
    aln_file = os.path.join(aln_dir, target_key + '.aln')
    # if 'X' in target_sequence:
    #     print(target_key)
    feature = target_feature(aln_file, target_sequence)
    return feature


# pconsc4 predicted contact map save in data/dataset/pconsc4
def target_to_graph(target_key, target_sequence, contact_dir, aln_dir):
    target_edge_index = []
    target_size = len(target_sequence)
    # contact_dir = 'data/' + dataset + '/pconsc4'
    contact_file = os.path.join(contact_dir, target_key + '.npy')
    contact_map = np.load(contact_file)
    contact_map += np.matrix(np.eye(contact_map.shape[0]))
    index_row, index_col = np.where(contact_map >= 0.5)
    for i, j in zip(index_row, index_col):
        target_edge_index.append([i, j])
    target_feature = target_to_feature(target_key, target_sequence, aln_dir)
    target_edge_index = np.array(target_edge_index)
    return target_size, target_feature, target_edge_index


# to judge whether the required files exist
def valid_target(key, dataset):
    contact_dir = 'data/' + dataset + '/pconsc4'
    aln_dir = 'data/' + dataset + '/aln'
    contact_file = os.path.join(contact_dir, key + '.npy')
    aln_file = os.path.join(aln_dir, key + '.aln')
    # print(contact_file, aln_file)
    if os.path.exists(contact_file) and os.path.exists(aln_file):
        return True
    else:
        return False


def data_to_csv(csv_file, datalist):
    with open(csv_file, 'w') as f:
        f.write('compound_iso_smiles,drug_smiles,target_sequence,target_key,affinity\n')
        for data in datalist:
            f.write(','.join(map(str, data)) + '\n')


def create_dataset_for_test(dataset):
    # load dataset
    dataset_path = 'data/' + dataset + '/'
    test_fold = json.load(open(dataset_path + 'folds/test_fold_setting1.txt'))
    ligands = json.load(open(dataset_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    # load contact and aln
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'
    msa_list = []
    contact_list = []
    for key in proteins:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contac_path, key + '.npy'))

    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    # smiles
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    # seqs
    for t in proteins.keys():
        prots.append(proteins[t])
        prot_keys.append(t)
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    valid_test_count = 0
    rows, cols = np.where(np.isnan(affinity) == False)
    rows, cols = rows[test_fold], cols[test_fold]
    temp_test_entries = []
    for pair_ind in range(len(rows)):
        # if the required files is not exist, then pass
        if not valid_target(prot_keys[cols[pair_ind]], dataset):
            continue
        ls = []
        ls += [drugs[rows[pair_ind]]]
        ls += [drug_smiles[rows[pair_ind]]]
        ls += [prots[cols[pair_ind]]]
        ls += [prot_keys[cols[pair_ind]]]
        ls += [affinity[rows[pair_ind], cols[pair_ind]]]
        temp_test_entries.append(ls)
        valid_test_count += 1
    csv_file = 'data/' + dataset + '_test.csv'
    data_to_csv(csv_file, temp_test_entries)
    print('dataset:', dataset)
    print('test entries:', len(test_fold), 'effective test entries', valid_test_count)

    compound_iso_smiles = drugs
    target_key = prot_keys

    # create smile graph
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g

    # count the number of  proteins with aln and contact files
    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    # 'data/davis_test.csv' or data/kiba_test.csv'
    df_test = pd.read_csv('data/' + dataset + '_test.csv')

    test_drugs, test_drugs_smiles, test_prot_keys, test_prot_seq, test_Y = list(
        df_test['compound_iso_smiles']), list(df_test['drug_smiles']), list(
        df_test['target_key']), list(df_test['target_sequence']), list(df_test['affinity'])
    test_proteins_vec = [seq_cat(t) for t in test_prot_seq]
    test_smiles_vec = [smi_cat(t) for t in test_drugs_smiles]
    test_drugs, test_prot_keys, test_proteins_vec, test_smiles_vec, test_Y = np.asarray(test_drugs), np.asarray(
        test_prot_keys), np.asarray(test_proteins_vec), np.asarray(test_smiles_vec), np.asarray(test_Y)
    test_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=test_drugs, target_key=test_prot_keys,
                               y=test_Y, smile_graph=smile_graph, target_graph=target_graph,
                               smiles_vec=test_smiles_vec, proteins_vec=test_proteins_vec)

    # test_drugs, test_prot_keys, test_Y = list(df_test['compound_iso_smiles']), list(df_test['target_key']), list(
    #     df_test['affinity'])
    # test_drugs, test_prot_keys, test_Y = np.asarray(test_drugs), np.asarray(test_prot_keys), np.asarray(test_Y)
    # test_dataset = DTADataset(root='data', dataset=dataset + '_test', xd=test_drugs, y=test_Y,
    #                           target_key=test_prot_keys, smile_graph=smile_graph, target_graph=target_graph)

    return test_dataset


def create_dataset_for_5folds(dataset, fold=0):
    # load dataset
    dataset_path = 'data/' + dataset + '/'
    train_fold_origin = json.load(open(dataset_path + 'folds/train_fold_setting1.txt'))
    train_fold_origin = [e for e in train_fold_origin]  # for 5 folds

    ligands = json.load(open(dataset_path + 'ligands_can.txt'), object_pairs_hook=OrderedDict)
    proteins = json.load(open(dataset_path + 'proteins.txt'), object_pairs_hook=OrderedDict)

    # with open(dataset_path + 'ligands_can.txt', 'r') as f:
    #     smiles =[i.split(":")[1] for i in f.read().strip().split(",")]
    # with open(dataset_path + 'proteins.txt', 'r') as f:
    #     protein = f.read().strip().split("\n")

    # load contact and aln
    msa_path = 'data/' + dataset + '/aln'
    contac_path = 'data/' + dataset + '/pconsc4'
    msa_list = []
    contact_list = []
    for key in proteins:
        msa_list.append(os.path.join(msa_path, key + '.aln'))
        contact_list.append(os.path.join(contac_path, key + '.npy'))

    # load train,valid and test entries
    train_folds = []
    valid_fold = train_fold_origin[fold]  # one fold
    for i in range(len(train_fold_origin)):  # other folds
        if i != fold:
            train_folds += train_fold_origin[i]

    affinity = pickle.load(open(dataset_path + 'Y', 'rb'), encoding='latin1')
    drugs = []
    prots = []
    prot_keys = []
    drug_smiles = []
    smiles_vec = {}
    proteins_vec = {}
    # smiles

    # smile_, smi_word_index, smi_embedding_matrix = smile_w2v_pad(ligands[d], 100, 100)
    for d in ligands.keys():
        # smile_, smi_word_index, smi_embedding_matrix = smile_w2v_pad(ligands[d], 100, 100)
        # smile_ = torch.tensor(smile_, dtype=)
        # smiles_vec[ligands[d]] = smile_
        # smiles_.append(ligands[d])
        # smile_vec, smi_word_index, smi_embedding_matrix = smile_w2v_pad(ligands[d], 100, 100)
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        # smiles_vec[lg] = smile_
        drugs.append(lg)
        drug_smiles.append(ligands[d])
    # smiles_vec, smi_word_index, smi_embedding_matrix = smile_w2v_pad(drug_smiles, 100, 100)
    # seqs

    for t in proteins.keys():
        # protein_, pro_word_index, pro_embedding_matrix = w2v_pad(proteins[t], 2000, 100)
        # proteins_vec[t] = protein_
        # proteins_.append(proteins[t])
        prots.append(proteins[t])
        prot_keys.append(t)
    # proteins_vec, pro_word_index, pro_embedding_matrix = w2v_pad(prots, 2000, 100)
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)

    opts = ['train', 'valid']
    valid_train_count = 0
    valid_valid_count = 0
    for opt in opts:
        if opt == 'train':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[train_folds], cols[train_folds]
            train_fold_entries = []
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):  # ensure the contact and aln files exists
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [drug_smiles[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                train_fold_entries.append(ls)
                valid_train_count += 1

            csv_file = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + opt + '.csv'
            data_to_csv(csv_file, train_fold_entries)
        elif opt == 'valid':
            rows, cols = np.where(np.isnan(affinity) == False)
            rows, cols = rows[valid_fold], cols[valid_fold]
            valid_fold_entries = []
            for pair_ind in range(len(rows)):
                if not valid_target(prot_keys[cols[pair_ind]], dataset):
                    continue
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [drug_smiles[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [prot_keys[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                valid_fold_entries.append(ls)
                valid_valid_count += 1

            csv_file = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + opt + '.csv'
            data_to_csv(csv_file, valid_fold_entries)
    print('dataset:', dataset)
    # print('len(set(drugs)),len(set(prots)):', len(set(drugs)), len(set(prots)))

    # entries with protein contact and aln files are marked as effiective
    print('fold:', fold)
    print('train entries:', len(train_folds), 'effective train entries', valid_train_count)
    print('valid entries:', len(valid_fold), 'effective valid entries', valid_valid_count)

    compound_iso_smiles = drugs
    target_key = prot_keys

    # smile_vec = []
    # create smile graph
    smile_graph = {}
    for smile in compound_iso_smiles:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    # print(smile_graph['CN1CCN(C(=O)c2cc3cc(Cl)ccc3[nH]2)CC1']) #for test

    # create target graph
    # print('target_key', len(target_key), len(set(target_key)))
    target_graph = {}
    for key in target_key:
        if not valid_target(key, dataset):  # ensure the contact and aln files exists
            continue
        g = target_to_graph(key, proteins[key], contac_path, msa_path)
        target_graph[key] = g

    # count the number of  proteins with aln and contact files

    print('effective drugs,effective prot:', len(smile_graph), len(target_graph))
    if len(smile_graph) == 0 or len(target_graph) == 0:
        raise Exception('no protein or drug, run the script for datasets preparation.')

    # 'data/davis_fold_0_train.csv' or data/kiba_fold_0__train.csv'
    train_csv = 'data/' + dataset + '_' + 'fold_' + str(fold) + '_' + 'train' + '.csv'
    df_train_fold = pd.read_csv(train_csv)
    train_drugs, train_drugs_smiles, train_prot_keys, train_prot_seq, train_Y = list(df_train_fold['compound_iso_smiles']), list(df_train_fold['drug_smiles']), list(
        df_train_fold['target_key']), list(df_train_fold['target_sequence']), list(df_train_fold['affinity'])
    train_proteins_vec = [seq_cat(t) for t in train_prot_seq]
    train_smiles_vec = [smi_cat(t) for t in train_drugs_smiles]
    # smiles_vec, smi_word_index, smi_embedding_matrix = smile_w2v_pad(train_drugs, 100, 100)
    train_drugs, train_prot_keys, train_proteins_vec, train_smiles_vec, train_Y = np.asarray(train_drugs), np.asarray(train_prot_keys), np.asarray(train_proteins_vec), np.asarray(train_smiles_vec), np.asarray(train_Y)
    train_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=train_drugs, target_key=train_prot_keys,
                                y=train_Y, smile_graph=smile_graph, target_graph=target_graph, smiles_vec=train_smiles_vec, proteins_vec=train_proteins_vec)

    df_valid_fold = pd.read_csv('data/' + dataset + '_' + 'fold_' + str(fold) + '_' + 'valid' + '.csv')
    valid_drugs, valid_drugs_smiles, valid_prots_keys, valid_prot_seq, valid_Y = list(df_valid_fold['compound_iso_smiles']), list(df_valid_fold['drug_smiles']),list(
        df_valid_fold['target_key']), list(df_valid_fold['target_sequence']), list(df_valid_fold['affinity'])
    valid_proteins_vec = [seq_cat(t) for t in valid_prot_seq]
    valid_smiles_vec = [smi_cat(t) for t in valid_drugs_smiles]
    valid_drugs, valid_prots_keys, valid_proteins_vec, valid_smiles_vec, valid_Y = np.asarray(valid_drugs), np.asarray(valid_prots_keys), np.asarray(valid_proteins_vec), np.asarray(valid_smiles_vec), np.asarray(
        valid_Y)
    valid_dataset = DTADataset(root='data', dataset=dataset + '_' + 'train', xd=valid_drugs,
                               target_key=valid_prots_keys, y=valid_Y, smile_graph=smile_graph,
                               target_graph=target_graph, smiles_vec=valid_smiles_vec, proteins_vec=valid_proteins_vec)
    return train_dataset, valid_dataset
