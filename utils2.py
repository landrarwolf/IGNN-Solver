import numpy as np
import scipy.sparse as sp
import torch

from normalization import row_normalize, aug_normalized_adjacency


# utils from AM-GCN

def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost


# def accuracy(output, labels):
#     labels = labels.max(1)[1]
#     preds = output.max(1)[1].type_as(labels)
#     correct = preds.eq(labels).double()
#     correct = correct.sum()
#     return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    
    return sparse_mx


def load_data(config):
    feature_path = np.loadtxt(config.feature_path, dtype=float)
    label_path = np.loadtxt(config.label_path, dtype=int)
    test = np.loadtxt(config.test_path, dtype=int)
    train = np.loadtxt(config.train_path, dtype=int)
    features = sp.csr_matrix(feature_path, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))
    
    idx_test = test.tolist()
    idx_train = train.tolist()
    
    idx_train = torch.LongTensor(idx_train)
    idx_test = torch.LongTensor(idx_test)
    
    label = torch.LongTensor(np.array(label_path))
    
    return features, label, idx_train, idx_test


def load_graph(dataset, config, Laplace_normalization=False):
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n),
                         dtype=np.float32)
    
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    
    sadj = sadj + sp.eye(sadj.shape[0])
    
    if Laplace_normalization is False:
        sadj = row_normalize(sadj)
    else:
        sadj = aug_normalized_adjacency(sadj, plus_I=False)
    
    nsadj = sparse_mx_to_torch_sparse_tensor(sadj)
    
    return nsadj
