from argparse import Namespace
import os.path as osp
from pathlib import Path

import torch
from torch_geometric.datasets import PPI
from torch_geometric.data import (InMemoryDataset, Data, DataLoader)
import torch_geometric.transforms as T

import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import eigsh  # For sparse matrices

class ProcessedDataset(InMemoryDataset):
    pass


def precompute_edge_label_and_reverse(dataset: InMemoryDataset):
    data_list = []
    for data in dataset:
        u, v = data.edge_index
        yu, yv = data.y[u], data.y[v]
        data.edge_labels = yu * dataset.num_classes + yv

        edge_dict = torch.sparse_coo_tensor(indices=data.edge_index, values=torch.arange(data.num_edges), size=(data.num_nodes, data.num_nodes)).to_dense()
        data.edge_index_reversed = edge_dict[v, u]

        data_list.append(data)

    new_data, new_slices = InMemoryDataset.collate(data_list)
    new_dataset = ProcessedDataset('.')
    new_dataset.data = new_data
    new_dataset.slices = new_slices
    return new_dataset

def compute_laplacian_positional_encoding(data, pe_dim):
    """
    Computes the Laplacian positional encoding for a given graph.

    Args:
        data (Data): The data object containing the graph.
        pe_dim (int): The number of Laplacian eigenvectors to use.

    Returns:
        data (Data): The data object with the positional encodings added.
    """
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = data.edge_attr

    if edge_weight is None:
        print("No edge weights provided, using uniform weights.")
        edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    # Create the adjacency matrix in scipy sparse format
    adj = to_scipy_sparse_matrix(edge_index, edge_attr=edge_weight, num_nodes=num_nodes)

    # Compute the normalized Laplacian
    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0.0
    deg_inv_sqrt_mat = sp.diags(deg_inv_sqrt)
    normalized_adj = deg_inv_sqrt_mat @ adj @ deg_inv_sqrt_mat

    laplacian = sp.eye(num_nodes) - normalized_adj

    # Compute the smallest non-trivial eigenvectors
    try:
        # Use eigsh for sparse symmetric matrices
        eigenvalues, eigenvectors = eigsh(laplacian, k=pe_dim+1, which='SM', tol=1e-2)
        # Exclude the first trivial eigenvector
        positional_encoding = torch.from_numpy(eigenvectors[:, 1:pe_dim+1]).float()
    except Exception as e:
        print(f"Error computing Laplacian PE: {e}")
        # Fall back to random positional encodings
        positional_encoding = torch.randn(num_nodes, pe_dim)

    # Normalize positional encodings
    positional_encoding = F.normalize(positional_encoding, p=2, dim=1)

    # Store the positional encodings in the data object
    data.pos_enc = positional_encoding

    # Concatenate positional encodings with node features
    data.x = torch.cat([data.x, data.pos_enc], dim=1)
    # data.num_features = data.x.size(1) 

    return data

def compute_laplacian_pe_dataset(dataset: InMemoryDataset, pe_dim: int):
    data_list = []
    for data in dataset:
        data = compute_laplacian_positional_encoding(data, pe_dim)
        data_list.append(data)
    new_data, new_slices = InMemoryDataset.collate(data_list)
    new_dataset = ProcessedDataset('.')
    new_dataset.data = new_data
    new_dataset.slices = new_slices
    return new_dataset


class BinaryPPI(PPI):
    def __init__(self, root, split, transform=None):
        super().__init__(root, split=split, transform=transform)

    @property
    def num_classes(self):
        return 2


def prepare_PPI(args: Namespace, path=osp.join('.', 'data', 'PPI')):
    gid, lid = map(int, args.dataset.split('-')[1:])
    assert gid in range(1, 21), f'gid should be in 1-20, got {gid}'
    assert lid in range(121), f'lid should be in 0-120, got {lid}'

    def transform(data):
        data.y = data.y[:, lid].long()
        return data

    train_dataset = BinaryPPI(path, split='train', transform=transform)[list(range(gid))]
    val_dataset = BinaryPPI(path, split='val', transform=transform)
    test_dataset = BinaryPPI(path, split='test', transform=transform)
    return train_dataset, val_dataset, test_dataset


class CitationDataset(InMemoryDataset):
    def __init__(self, root=None, split='train', transform=None, pre_transform=None, pre_filter=None):
        assert split in ['train', 'val', 'test']
        super(CitationDataset, self).__init__(root, transform, pre_transform, pre_filter)

        saved_data = torch.load(root)
        
        self.data = Data(edge_index=saved_data['{}_e'.format(split)], x=saved_data['{}_x'.format(split)], y=saved_data['{}_y'.format(split)])
        num_nodes = self.data.x.size(0)
        num_edges = self.data.edge_index.size(1)
        self.slices = {
            'x': torch.LongTensor([0, num_nodes]), 
            'y': torch.LongTensor([0, num_nodes]),
            'edge_index': torch.LongTensor([0, num_edges])
        }


class BatchedCitationDataset(InMemoryDataset):
    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None):
        super(BatchedCitationDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.data = torch.load(root)
        
        num_nodes = self.data.x.size(0)
        num_edges = self.data.edge_index.size(1)
        self.slices = {
            'x': torch.LongTensor([0, num_nodes]), 
            'y': torch.LongTensor([0, num_nodes]),
            'edge_index': torch.LongTensor([0, num_edges]),
            'batch': torch.LongTensor([0, num_edges])
        }


def prepare_dblp(args: Namespace):
    path = osp.join('.', 'data', 'Citation', 'dblp.pkl')

    train_dataset = CitationDataset(root=path, split='train')  #, transform=T.NormalizeFeatures())
    val_dataset = CitationDataset(root=path, split='val')  #, transform=T.NormalizeFeatures())
    test_dataset = CitationDataset(root=path, split='test')  #, transform=T.NormalizeFeatures())
    return train_dataset, val_dataset, test_dataset


def prepare_dataloaders(args: Namespace):
    """
    Prepare train, valid and test dataloaders.
    Args:
        args: Arguments from the command line. See config.py.

    Returns:
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        test_loader: DataLoader for test set.
    """
    if args.dataset.startswith('ppi-'):
        train_dataset, val_dataset, test_dataset = map(precompute_edge_label_and_reverse, prepare_PPI(args))
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    elif args.dataset == 'dblp':
        train_dataset, val_dataset, test_dataset = map(precompute_edge_label_and_reverse, prepare_dblp(args))
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    elif args.dataset in ['cora', 'citeseer', 'pubmed']:
        root = Path('./data/Citation')
        train_loader = BatchedCitationDataset(root=root / f'{args.dataset}_train.pt')
        val_loader = BatchedCitationDataset(root=root / f'{args.dataset}_val.pt')
        test_loader = BatchedCitationDataset(root=root / f'{args.dataset}_test.pt')
        train_loader, val_loader, test_loader = map(precompute_edge_label_and_reverse, (train_loader, val_loader, test_loader))
        
        # Compute and add Laplacian PE if enabled
        if args.use_laplacian_pe:
            train_loader = compute_laplacian_pe_dataset(train_loader, args.pe_dim)
            val_loader = compute_laplacian_pe_dataset(val_loader, args.pe_dim)
            test_loader = compute_laplacian_pe_dataset(test_loader, args.pe_dim)

        def _set_dataset_attr(loader):
            loader.dataset = loader
            return loader
        train_loader, val_loader, test_loader = map(_set_dataset_attr, (train_loader, val_loader, test_loader))
    else:
        raise NotImplementedError(f'Dataset {args.dataset} not supported.')
    return train_loader, val_loader, test_loader
