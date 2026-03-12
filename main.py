from __future__ import division
from __future__ import print_function

import warnings

from scipy.sparse.linalg import inv, spsolve
from sklearn.neighbors import NearestNeighbors
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
torch.set_printoptions(threshold=1000000)
import torch.nn as nn
import numpy as np
import random
np.set_printoptions(threshold=10e6)
import pandas as pd
import scipy.sparse as sp
from argparse import ArgumentParser, Namespace
import os
import time
from typing import Union, Tuple
import json

np.set_printoptions(threshold=np.inf)

from utils import get_roc_score
from utils import save_checkpoint, load_checkpoint
from utils import create_logger, makedirs
from model import DDI


# from features import get_features_generator, get_available_features_generators

def parse_args():
    parser = ArgumentParser()
    # data
    parser.add_argument('--dataset', type=str, default='zhang',
                        choices=['zhang', 'ChChMiner', 'DeepDDI'])
    parser.add_argument('--train_data_path', type=str, default='train.csv')
    parser.add_argument('--valid_data_path', type=str, default='valid.csv')
    parser.add_argument('--test_data_path', type=str, default='test.csv')

    # training
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--L2', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--activation', type=str, default='ELU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')

    parser.add_argument('--dnn_dropout', type=float, default=0.1, help='Dropout rate of DNN.')

    parser.add_argument('--hidden_dim', type=int, default=32, help='Output dim')
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=1)

    # store
    parser.add_argument('--save_dir', type=str, default='./model_save',
                        help='Directory where model checkpoints will be saved')
    parser.add_argument('--quiet', action='store_true', default=False,
                        help='Skip non-essential print statements')
    # cross-attention 参数
    parser.add_argument('--cross_n_heads', type=int, default=4,
                        help='number of heads for cross-attention')
    parser.add_argument('--cross_dropout', type=float, default=0.1,
                        help='dropout for cross-attention')

    args = parser.parse_args()
    # config
    if args.dataset == 'zhang':
        args.weight_decay = 0.01

    return args


def seed_everything():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    torch.backends.cudnn.deterministic = True


def load_vocab(filepath: str):
    filepath = os.path.join('data', filepath, 'drug_list.csv')
    df = pd.read_csv(filepath, index_col=False)
    smiles2id = {smiles: idx for smiles, idx in zip(df['smiles'], range(len(df)))}
    return smiles2id

def load_feature(dataset: str):
    filepath = os.path.join('data', dataset, 'chem_Jacarrd_sim.csv')
    df = pd.read_csv(filepath, index_col=0)
    features = df.values
    return torch.tensor(features).float()

def load_csv_data(filepath: str, smiles2id: dict, is_train_file: bool = True) \
        -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(filepath, index_col=False)

    edges = []
    edges_false = []
    for row_id, row in df.iterrows():
        row_dict = dict(row)
        smiles_1 = row_dict['smiles_1']
        smiles_2 = row_dict['smiles_2']
        if smiles_1 in smiles2id.keys() and smiles_2 in smiles2id.keys():
            idx_1 = smiles2id[smiles_1]
            idx_2 = smiles2id[smiles_2]
            label = int(row_dict['label'])
        else:
            continue
        if label > 0:
            edges.append((idx_1, idx_2))
            edges.append((idx_2, idx_1))
        else:
            edges_false.append((idx_1, idx_2))
            edges_false.append((idx_2, idx_1))
    if is_train_file:
        edges = np.array(edges, dtype=int)
        edges_false = np.array(edges_false, dtype=int)
        return edges, edges_false
    else:
        edges = np.array(edges, dtype=int)
        edges_false = np.array(edges_false, dtype=int)
        return edges, edges_false


def build_diffusion_graph(adj_matrix, t=1):
    """
    构建扩散图。
    :param adj_matrix: 输入图的邻接矩阵（稀疏矩阵格式）
    :param t: 扩散时间参数，控制扩散距离
    :return: 扩散图邻接矩阵（稀疏矩阵格式）
    """
    # 计算度矩阵
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-8))  # 避免除零错误

    # 计算归一化拉普拉斯矩阵
    laplacian = sp.eye(adj_matrix.shape[0]) - d_inv_sqrt @ adj_matrix @ d_inv_sqrt

    # 计算扩散核：exp(-t * L)，L为拉普拉斯矩阵
    diffusion_matrix = sp.eye(adj_matrix.shape[0]) - t * laplacian
    diffusion_adj = diffusion_matrix.maximum(0)  # 保证非负性

    return diffusion_adj


def build_diffusion_graph_ppr(adj_matrix, alpha=0.2):
    """
    构建基于PPR的扩散图。

    :param adj_matrix: 输入图的邻接矩阵（稀疏矩阵格式）
    :param alpha: PPR的参数
    :return: 扩散图邻接矩阵（稀疏矩阵格式）
    """
    # 计算度矩阵并进行归一化
    degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(degrees + 1e-8))  # 避免除零错误
    norm_adj = d_inv_sqrt @ adj_matrix @ d_inv_sqrt  # 归一化邻接矩阵

    # 构建单位矩阵
    identity_matrix = sp.eye(adj_matrix.shape[0])  # 创建单位矩阵

    # 使用稀疏矩阵的逆来计算PPR矩阵
    ppr_matrix = alpha * spsolve(identity_matrix - (1 - alpha) * norm_adj, np.ones(adj_matrix.shape[0]))

    # 将PPR值转换为稀疏矩阵
    ppr_sparse = sp.diags(ppr_matrix)

    return ppr_sparse

def get_diff_adj(adj, alpha=0.2):
    # 对称归一化
    adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))

    # 构建PPR矩阵: (I - α * P)^(-1)
    I = sp.eye(adj_norm.shape[0])
    diff_matrix = I - alpha * adj_norm

    # 求解得到完整的扩散邻接矩阵
    diff_matrix = diff_matrix.tocsc()
    solver = sp.linalg.splu(diff_matrix)
    diff_adj = solver.solve(I.toarray()).T

    # 应该返回完整的扩散矩阵，而不是仅对角线
    return sp.csr_matrix(diff_adj)



def sparse_to_tuple(sparse_mx: sp.dia_matrix) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def sparse_mx_to_torch_sparse_tensor(sparse_mx) \
        -> torch.sparse.torch.sparse_coo_tensor:
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.torch.sparse_coo_tensor(indices, values, shape)



def normalize_adj(adj: sp.csr_matrix) -> sp.coo_matrix:
    """
    计算 \hat{A} = D^{-1/2} (A + I) D^{-1/2}
    - 加自环，增强特征保持与训练稳定性
    - 稳定处理 0 度节点
    - 返回 coo，兼容现有 sparse 转换与调用
    """
    if not sp.isspmatrix_coo(adj):
        adj = adj.tocoo()
    # 加自环
    adj_with_self = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format='coo')
    # 度向量（按行求和）
    rowsum = np.array(adj_with_self.sum(1)).flatten()
    # D^{-1/2}，对 0 度置 0 避免 inf
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[rowsum == 0] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    # 归一化
    adj_norm = D_inv_sqrt @ adj_with_self @ D_inv_sqrt
    return adj_norm.tocoo()

# def normalize_adj(adj: sp.csr_matrix) -> sp.coo_matrix:
#     adj = sp.coo_matrix(adj)
#     # eliminate self-loop
#     adj_ = adj
#     rowsum = np.array(adj_.sum(0))
#     rowsum_power = []
#     for i in rowsum:
#         for j in i:
#             if j != 0:
#                 j_power = np.power(j, -0.5)
#                 rowsum_power.append(j_power)
#             else:
#                 j_power = 0
#                 rowsum_power.append(j_power)
#     rowsum_power = np.array(rowsum_power)
#     degree_mat_inv_sqrt = sp.diags(rowsum_power)
#
#     adj_norm = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
#     return adj_norm
#

def load_data(args: Namespace, smiles2idx: dict = None):
    assert smiles2idx is not None
    num_nodes = len(smiles2idx)
    train_edges, train_edges_false = load_csv_data(os.path.join('data', args.dataset, args.train_data_path), smiles2idx,
                                                   is_train_file=True)
    val_edges, val_edges_false = load_csv_data(os.path.join('data', args.dataset, args.valid_data_path), smiles2idx,
                                               is_train_file=False)
    test_edges, test_edges_false = load_csv_data(os.path.join('data', args.dataset, args.test_data_path), smiles2idx,
                                                 is_train_file=False)

    all_edges = np.concatenate([train_edges, val_edges, test_edges], axis=0)
    data = np.ones(all_edges.shape[0])

    adj = sp.csr_matrix((data, (all_edges[:, 0], all_edges[:, 1])),
                        shape=(num_nodes, num_nodes))
    data_train = np.ones(train_edges.shape[0])
    data_train_false = np.ones(train_edges_false.shape[0])

    adj_train = sp.csr_matrix((data_train, (train_edges[:, 0], train_edges[:, 1])),
                              shape=(num_nodes, num_nodes))
    adj_train_false = sp.csr_matrix((data_train_false, (train_edges_false[:, 0], train_edges_false[:, 1])), \
                                    shape=(num_nodes, num_nodes))

    return adj, adj_train, adj_train_false, val_edges, val_edges_false, test_edges, test_edges_false


args = parse_args()
# Create a unique run id and logger that writes under ./log/train/<dataset>/<run_id>/
run_id = time.strftime('%Y%m%d_%H%M%S')
save_dir_for_logger = os.path.join('log', 'train', args.dataset, run_id)
logger = create_logger(name='train', save_dir=save_dir_for_logger, quiet=args.quiet, run_id=run_id)
seed_everything()
args.cuda = True if torch.cuda.is_available() else False
# args.cuda = False
if args.cuda:
    torch.cuda.set_device(args.gpu)

def multi_scale_ppr(adj, alphas=[0.1, 0.3, 0.5]):
    """多尺度PPR扩散，捕获不同阶邻域信息"""
    diffusion_graphs = []
    for alpha in alphas:
        diff_adj = get_diff_adj(adj, alpha)
        diffusion_graphs.append(diff_adj)
    return diffusion_graphs  # 返回列表：[diff_adj_0.1, diff_adj_0.3, diff_adj_0.5]

def main():
    run_start = time.time()
    # Save per-run metrics under the dataset-specific folder created for the logger
    logs_dir = os.path.join('log', 'train', args.dataset, run_id)
    # --------------------------------------load data--------------------

    original_adj, adj_train, adj_train_false, val_edges, val_edges_false, test_edges, test_edges_false = \
        load_data(args, smiles2idx=load_vocab(args.dataset))
    feature = load_feature(args.dataset)
    # adj_diff = get_diff_adj(adj_train, 0.5)
    adj_diff_list = multi_scale_ppr(adj_train, alphas=[0.1, 0.3, 0.5])  # 生成多尺度扩散图
    adj_diff_norm_list = [normalize_adj(diff_adj) for diff_adj in adj_diff_list]
    adj_diff_norm_tensors = [sparse_mx_to_torch_sparse_tensor(adj_norm) for adj_norm in adj_diff_norm_list]
    args.feature_dim = feature.shape[1]
    args.drug_nums = feature.shape[0]

    # ---------------------------log info-------------------------
    num_nodes = original_adj.shape[0]
    num_edges = original_adj.nnz

    logger.info('Dataset: {}'.format(args.dataset))
    logger.info('Number of nodes: {}, number of edges: {}'.format(num_nodes, num_edges))

    features_nonzero = 0
    args.features_nonzero = features_nonzero

    # input for model
    num_edges_w = adj_train.sum()
    num_nodes_w = adj_train.shape[0]
    args.num_edges_w = num_edges_w
    pos_weight = float(num_nodes_w ** 2 - num_edges_w) / num_edges_w

    adj_norm = normalize_adj(adj_train)
    # adj_diff_norm = normalize_adj(adj_diff)

    adj_label = adj_train
    adj_mask = pos_weight * adj_train.toarray() + adj_train_false.toarray()
    adj_mask = torch.flatten(torch.Tensor(adj_mask))

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_norm)

    # adj_diff_norm = sparse_mx_to_torch_sparse_tensor(adj_diff_norm)

    adj_label = sparse_mx_to_torch_sparse_tensor(adj_label)
    lbl = torch.cat((torch.ones(num_nodes), torch.ones(num_nodes)))

    model = DDI(args)
    if args.cuda:
        adj_norm = adj_norm.cuda()
        adj_diff_norm_tensors = [tensor.cuda() for tensor in adj_diff_norm_tensors]
        adj_label = adj_label.cuda()
        adj_mask = adj_mask.cuda()
        model = model.cuda()
        feature = feature.cuda()

    loss_function_BCE = nn.BCEWithLogitsLoss(reduction='none')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    best_roc, best_epoch = 0, 0

    for epoch in range(args.epochs):
        t = time.time()

        model.train()
        optimizer.zero_grad()

        preds_out, MSEloss, _ = model(feature, adj_norm, adj_diff_norm_tensors)#feature杰卡德相似度矩阵。adj_normDDI原始图。adj_diff_normPPR扩散图。
        # print(model)
        preds = preds_out.view(-1)
        labels = adj_label.to_dense().view(-1)
        BCEloss = torch.mean(loss_function_BCE(preds, labels) * adj_mask)

        total_loss = BCEloss + MSEloss
        # total_loss = BCEloss

        model.eval()
        preds_out, _, _ = model(feature, adj_norm, adj_diff_norm_tensors)
        roc_curr, ap_curr, f1_curr, acc_curr = get_roc_score(
            preds_out, val_edges, val_edges_false
        )
        logger.info(
            'Epoch: {} train_loss= {:.5f} val_roc= {:.5f} val_ap= {:.5f}, val_f1= {:.5f}, val_acc={:.5f}, time= {:.5f} '.format(
                epoch + 1, total_loss, roc_curr, ap_curr, f1_curr, acc_curr, time.time() - t
            ))
        if roc_curr > best_roc and epoch > 150:
            best_roc = roc_curr
            best_epoch = epoch + 1
            if args.save_dir:
                # 确保目录存在后再保存模型
                best_model_path = os.path.join(args.save_dir, args.dataset, f'best_model.pt')
                makedirs(best_model_path, isfile=True)
                save_checkpoint(best_model_path, model, args)

        # update parameters
        total_loss.backward()
        optimizer.step()

    logger.info('Optimization Finished!')

    # 保存最终模型以防最佳模型未被保存
    if args.save_dir:
        final_model_path = os.path.join(args.save_dir, args.dataset, f'final_model.pt')
        # 确保目录存在后再保存模型
        makedirs(final_model_path, isfile=True)
        save_checkpoint(final_model_path, model, args)

    # 尝试加载最佳模型，如果不存在则加载最终模型
    model_path = os.path.join(args.save_dir, args.dataset, 'best_model.pt')
    final_model_path = os.path.join(args.save_dir, args.dataset, 'final_model.pt')
    
    if os.path.exists(model_path):
        model = load_checkpoint(model_path, cuda=args.cuda, logger=logger)
    elif os.path.exists(final_model_path):
        model = load_checkpoint(final_model_path, cuda=args.cuda, logger=logger)
    else:
        logger.warning("No model checkpoint found, using current model")

    model.eval()
    preds_out, _, emb = model(feature, adj_norm, adj_diff_norm_tensors)
    roc_score, ap_score, f1_score, acc_score = get_roc_score(
        preds_out, test_edges, test_edges_false, test=True)
    # 创建 embeddings 文件夹（如果不存在）
    embeddings_dir = 'embeddings'
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)
    
    # 保存嵌入和测试边到 embeddings 文件夹
    torch.save(emb, os.path.join(embeddings_dir, f'{args.dataset}_drug_embeddings.pt'))
    torch.save(test_edges, os.path.join(embeddings_dir, f'{args.dataset}_test_edges.pt'))
    torch.save(test_edges_false, os.path.join(embeddings_dir, f'{args.dataset}_test_edges_false.pt'))
    logger.info('Dataset: {}'.format(args.dataset))
    logger.info('BEST  MODEL!')
    logger.info(f'Model best_val_roc = {best_roc:.6f} on epoch {best_epoch}')
    logger.info('Test ROC score: {:.5f}'.format(roc_score))
    logger.info('Test AP score: {:.5f}'.format(ap_score))
    logger.info('Test F1 score: {:.5f}'.format(f1_score))
    logger.info('Test ACC score: {:.5f}'.format(acc_score))
    logger.info('{:.4f}\t{:.4f}\t{:.4f}'.format(roc_score, ap_score, f1_score))

    # Save per-run metrics to the log folder
    try:
        os.makedirs(logs_dir, exist_ok=True)
        metrics = {
            'dataset': args.dataset,
            'best_val_roc': float(best_roc),
            'best_epoch': int(best_epoch),
            'test': {
                'roc': float(roc_score),
                'ap': float(ap_score),
                'f1': float(f1_score),
                'acc': float(acc_score)
            },
            'run_id': run_id,
            'duration_sec': float(time.time() - run_start),
            'args': {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in vars(args).items()}
        }
        with open(os.path.join(logs_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info('Saved metrics to %s', os.path.join(logs_dir, 'metrics.json'))
    except Exception as e:
        logger.warning('Failed to write metrics.json: %s', e)


if __name__ == '__main__':
    main()
