from argparse import Namespace
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.nn.parameter import Parameter

class DDI(nn.Module):
    def __init__(self, args):
        super(DDI, self).__init__()
        self.feat_encoder = FeatureEncoder(args)
        self.graph_encoder = GraphEncoder(args)
        self.activation = nn.ELU() if args.activation == 'ELU' else nn.ReLU()
        self.tau = args.tau
        self.multi_head_att_fusion = GlobalFeatureAttention(hidden_dim=args.hidden_dim * 3, num_heads=4)
        self.mse_loss_fn = nn.MSELoss()
        self.dnn_dropout_rate = args.dnn_dropout
        self.predictor = nn.Sequential(
            nn.Linear(args.hidden_dim * 3, args.drug_nums),
            nn.Dropout(args.dnn_dropout),
            nn.BatchNorm1d(args.drug_nums),
            self.activation,
            nn.Linear(args.drug_nums, args.drug_nums)
        )
        self.sigmoid = nn.Sigmoid()

        # 新增：两层双向 Cross\-Attention（可调层数或复用参数）
        # 使用 d_model = hidden_dim 以便保持维度一致，head 数可调整
        self.cross_att_1 = CrossViewCrossAttention(
            d=args.hidden_dim, d_model=args.hidden_dim,
            n_heads=args.cross_n_heads, dropout=args.cross_dropout
        )
        self.cross_att_2 = CrossViewCrossAttention(
            d=args.hidden_dim, d_model=args.hidden_dim,
            n_heads=args.cross_n_heads, dropout=args.cross_dropout
        )
        self.num_scales = 3  # 多尺度数量
        self.scale_fusion = nn.Linear(self.num_scales, 1)  # 可学习权重融合多尺度图

    def build_knn_graph(self, features, k=10):
        features = features.cpu().detach().numpy()
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(features)
        distances, indices = nbrs.kneighbors(features)
        num_nodes = features.shape[0]
        adj_matrix = sp.lil_matrix((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(1, k + 1):
                adj_matrix[i, indices[i, j]] = 1
                adj_matrix[indices[i, j], i] = 1
        adj_knn_norm = normalize_adj(adj_matrix.tocsr())
        adj_knn_norm = sparse_mx_to_torch_sparse_tensor(adj_knn_norm)
        return adj_knn_norm.cuda()


    # def build_knn_graph(self, features: torch.Tensor, k: int = 10, batch_size: int = 4096, symmetric: bool = True):
    #     """
    #     在 GPU 上基于特征构建 kNN 稀疏图并进行对称归一化。
    #     - features: (N, D) 的特征张量（建议已在 GPU）
    #     - k: 每个点的近邻数量（不含自身）
    #     - batch_size: 分批计算 cdist 以降低显存峰值
    #     - symmetric: 是否构建无向图（加入反向边）
    #     返回：torch.sparse_coo_tensor（与 features 同设备）
    #     """
    #     assert features.dim() == 2, "features must be 2D"
    #     device = features.device
    #     dtype = features.dtype
    #     N = features.size(0)
    #     k = min(k, max(N - 1, 1))  # 至多取 N-1 个近邻
    #
    #     # 分批计算到全体样本的距离，取 top-k 最近邻（排除自环）
    #     all_indices = []
    #     for start in range(0, N, batch_size):
    #         end = min(start + batch_size, N)
    #         chunk = features[start:end]  # (B, D)
    #
    #         # 欧氏距离（更稳健）；也可替换为余弦相似度的负值以近似
    #         dist = torch.cdist(chunk, features)  # (B, N)
    #
    #         # 排除自身：把对角置为 +inf（仅对齐到原索引范围）
    #         bsz = end - start
    #         row_idx = torch.arange(bsz, device=device)
    #         col_idx = torch.arange(start, end, device=device)
    #         dist[row_idx, col_idx] = float("inf")
    #
    #         # 取最小的 k 个距离对应的列索引
    #         knn_idx = torch.topk(dist, k=k, largest=False, dim=1).indices  # (B, k)
    #         all_indices.append(knn_idx)
    #
    #     knn_idx = torch.cat(all_indices, dim=0)  # (N, k)
    #
    #     # 构建 COO 边集
    #     row = torch.arange(N, device=device).unsqueeze(1).expand(N, k).reshape(-1)
    #     col = knn_idx.reshape(-1)
    #     val = torch.ones_like(row, dtype=dtype, device=device)
    #
    #     indices = torch.stack([row, col], dim=0)
    #     values = val
    #
    #     if symmetric:
    #         # 加入反向边，随后合并
    #         indices_sym = torch.stack([col, row], dim=0)
    #         values_sym = val
    #         indices = torch.cat([indices, indices_sym], dim=1)
    #         values = torch.cat([values, values_sym], dim=0)
    #
    #     adj_knn = torch.sparse_coo_tensor(
    #         indices, values, size=(N, N), device=device
    #     ).coalesce()
    #
    #     # 对称归一化（GPU）
    #     adj_knn_norm = normalize_adj_torch(adj_knn)
    #     return adj_knn_norm
    #
    def graph_encoder_single(self, x, adj):
        h1 = self.activation(self.graph_encoder.gnn1(x, adj))
        h2 = self.activation(self.graph_encoder.gnn2(h1, adj))
        h3 = self.activation(self.graph_encoder.gnn3(h2, adj))
        return h3


    def forward(self, x, adj, adj_diff_list):
        z1, z2, z3, x_d = self.feat_encoder(x)#z1:(N, hidden*4), z2:(N, hidden*2), z3:(N, hidden) x_d:(N, feature_dim)
        f_d = torch.cat((z1, z2, z3), dim=-1)  # (N, hidden_dim * 7)
        adj_knn = self.build_knn_graph(f_d)

        # 对每个尺度的扩散图用相同的单图编码器编码，得到多尺度特征列表
        f_diff_list = []
        for adj_diff in adj_diff_list:
            f_diff_scale = self.graph_encoder_single(f_d, adj_diff)  # (N, hidden_dim)
            f_diff_list.append(f_diff_scale)

        # 可学习融合多尺度 f_diff
        f_diff_stacked = torch.stack(f_diff_list, dim=-1)  # (N, hidden_dim, num_scales)
        weights = torch.softmax(self.scale_fusion(f_diff_stacked.mean(dim=1)), dim=-1)  # (N, num_scales)
        f_diff_fused = (f_diff_stacked * weights.unsqueeze(1)).sum(dim=-1)  # (N, hidden_dim)

        # 使用其中一个稀疏 adj_diff（传入的列表）作为 graph_encoder 的 adj_diff 参数以获得 f_t 和 f_knn
        # 注意：graph_encoder 返回的第三项也是基于该单尺度 adj_diff 的特征，我们与多尺度融合特征合并
        f_t, f_knn, f_diff_single = self.graph_encoder(f_d, adj, adj_knn, adj_diff_list[0])

        # 将单尺度图编码得到的 f_diff_single 与多尺度融合特征做简单平均以融合信息
        f_diff_feat = (f_diff_single + f_diff_fused) / 2.0

        mse_loss = self.mse_loss_fn(x_d, x)

        # --- 双向 Cross\-Attention 层 1 ---
        # 方向一：Q = f_t, KV = [f_knn, f_diff_feat] -> 输出 (N,1,d) -> fused_t
        fused_t_1 = self.cross_att_1(f_t, torch.stack([f_knn, f_diff_feat], dim=1)).squeeze(1)  # (N, d)
        # 方向二：Q = [f_knn, f_diff], KV = f_t -> 输出 (N,2,d) 对应 knn/diff
        fused_kd_1 = self.cross_att_1(torch.stack([f_knn, f_diff_feat], dim=1), f_t.unsqueeze(1))  # (N,2,d)
        fused_knn_1 = fused_kd_1[:, 0, :]
        fused_diff_1 = fused_kd_1[:, 1, :]

        # 残差融合
        f_t_1 = f_t + fused_t_1
        f_knn_1 = f_knn + fused_knn_1
        f_diff_1 = f_diff_feat + fused_diff_1

        # --- 双向 Cross\-Attention 层 2 （堆叠，增强交互）---
        fused_t_2 = self.cross_att_2(f_t_1, torch.stack([f_knn_1, f_diff_1], dim=1)).squeeze(1)
        fused_kd_2 = self.cross_att_2(torch.stack([f_knn_1, f_diff_1], dim=1), f_t_1.unsqueeze(1))
        fused_knn_2 = fused_kd_2[:, 0, :]
        fused_diff_2 = fused_kd_2[:, 1, :]

        # 再次残差融合
        f_t_final = f_t_1 + fused_t_2
        f_knn_final = f_knn_1 + fused_knn_2
        f_diff_final = f_diff_1 + fused_diff_2

        # 拼接回原先的三路表示 (N, 3 * hidden_dim)
        emb = torch.cat((f_t_final, f_knn_final, f_diff_final), dim=-1)

        # 可沿用原有的 multi_head_att_fusion（对三路融合结果再做一次全局注意力与归一化）
        emb = self.multi_head_att_fusion(emb)

        pred = self.sigmoid(self.predictor(emb))
        return pred, mse_loss, emb


class CrossViewCrossAttention(nn.Module):
    """
    Cross\-attention between query tokens and key/value tokens for each node independently.
    Q: (N, qV, d) or (N, d)
    KV: (N, kvV, d) or (N, d)
    返回与 Q 对应的输出 tokens，shape=(N, qV, d)
    """

    def __init__(self, d, d_model=None, n_heads=4, dropout=0.1):
        super().__init__()
        self.d = d
        self.d_model = d_model if d_model is not None else d
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.d_k = self.d_model // self.n_heads

        self.Wq = nn.Linear(d, self.d_model)
        self.Wk = nn.Linear(d, self.d_model)
        self.Wv = nn.Linear(d, self.d_model)

        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.fuse = nn.Linear(self.d_model, d)  # map back to original d per token
        self.layernorm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, KV):
        # normalize shapes
        if Q.dim() == 2:
            Q = Q.unsqueeze(1)  # (N, 1, d)
        if KV.dim() == 2:
            KV = KV.unsqueeze(1)  # (N, 1, d)

        N, qV, _ = Q.shape
        _, kvV, _ = KV.shape

        Qp = self.Wq(Q)  # (N, qV, d_model)
        Kp = self.Wk(KV)  # (N, kvV, d_model)
        Vp = self.Wv(KV)  # (N, kvV, d_model)

        def reshape_heads(x):
            # x: (N, V, d_model) -> (N, n_heads, V, d_k)
            N, V, _ = x.shape
            return x.view(N, V, self.n_heads, self.d_k).permute(0, 2, 1, 3)

        Qh = reshape_heads(Qp)  # (N, h, qV, d_k)
        Kh = reshape_heads(Kp)  # (N, h, kvV, d_k)
        Vh = reshape_heads(Vp)  # (N, h, kvV, d_k)

        # attention over view/token dimension (keys dimension)
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.d_k ** 0.5)  # (N, h, qV, kvV)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out_heads = torch.matmul(attn, Vh)  # (N, h, qV, d_k)
        out = out_heads.permute(0, 2, 1, 3).contiguous().view(N, qV, self.d_model)  # (N, qV, d_model)

        out = self.out_proj(out)
        out = out + Qp  # residual in d_model space
        out = self.layernorm(out)
        out = self.fuse(out)  # (N, qV, d)
        return out  # (N, qV, d)


class FeatureEncoder(nn.Module):
    def __init__(self, args):
        super(FeatureEncoder, self).__init__()
        self.activation = nn.ELU() if args.activation == 'ELU' else nn.ReLU()
        self.hidden1 = nn.Sequential(
            nn.Linear(args.feature_dim, args.hidden_dim * 4),
            nn.Dropout(args.dnn_dropout),
            nn.BatchNorm1d(args.hidden_dim * 4),
            self.activation,
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(args.hidden_dim * 4, args.hidden_dim * 2),
            nn.Dropout(args.dnn_dropout),
            nn.BatchNorm1d(args.hidden_dim * 2),
            self.activation,
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, args.hidden_dim),
            nn.Dropout(args.dnn_dropout),
            nn.BatchNorm1d(args.hidden_dim),
            self.activation,
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim * 2),
            self.activation,

            nn.Linear(args.hidden_dim * 2, args.hidden_dim * 4),
            self.activation,

            nn.Linear(args.hidden_dim * 4, args.feature_dim),  # 输出与输入维度一致
            self.activation,
        )

    def forward(self, x):
        z1 = self.hidden1(x)
        z2 = self.hidden2(z1)
        z3 = self.hidden3(z2)
        x_ = self.decoder(z3)
        return z1, z2, z3, x_


class GraphEncoder(nn.Module):
    def __init__(self, args: Namespace):
        super(GraphEncoder, self).__init__()
        self.dropout = nn.Dropout(args.dnn_dropout)
        self.gnn1 = GraphConvolution(in_features=args.hidden_dim * 7, out_features=args.hidden_dim * 4)
        self.gnn2 = GraphConvolution(in_features=args.hidden_dim * 4, out_features=args.hidden_dim * 2)
        self.gnn3 = GraphConvolution(in_features=args.hidden_dim * 2, out_features=args.hidden_dim)
        self.activation = nn.ELU() if args.activation == 'ELU' else nn.ReLU()

    def forward(self, x, adj, adj_knn, adj_diff):
        # GNN-1
        h1 = self.activation(self.gnn1(x, adj))
        hk1 = self.activation(self.gnn1(x, adj_knn))
        hd1 = self.activation(self.gnn1(x, adj_diff))

        # GNN-2
        h2 = self.activation(self.gnn2(h1, adj))
        hk2 = self.activation(self.gnn2(hk1, adj_knn))
        hd2 = self.activation(self.gnn2(hd1, adj_diff))

        # GNN-2
        h3 = self.activation(self.gnn3(h2, adj))
        hk3 = self.activation(self.gnn3(hk2, adj_knn))
        hd3 = self.activation(self.gnn3(hd2, adj_diff))

        # return torch.cat((h1, h2, h3), dim=-1), torch.cat((hk1, hk2, hk3), dim=-1), torch.cat((hd1, hd2, hd3), dim=-1)
        return h3, hk3, hd3


class GraphConvolution(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0)

    def forward(self, inputs: torch.Tensor, adj: torch.sparse.FloatTensor) -> torch.Tensor:
        support = torch.mm(inputs, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GAT(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):  # ()
        Wh = torch.mm(h, self.W)
        e = self._prepare_attentional_mechanism_input(Wh)

        zero_vec = -9e15 * torch.ones_like(e)
        adj = adj.to_dense()
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)  #
        h_prime = torch.matmul(attention, Wh)  #
        # return h_prime, attention
        return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])  # (572,1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.t()
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GlobalFeatureAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q/K/V 投影（每个头独立）
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # 层归一化
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        # 升维: (B, D) → (B, 1, D)
        x_seq = x.unsqueeze(1)

        # 生成 Q/K/V (每个头独立)
        q = self.q_proj(x_seq).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, C)
        k = self.k_proj(x_seq).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, C)
        v = self.v_proj(x_seq).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, C)

        # 注意力计算 (单元素序列)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, 1, 1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, 1, C)

        # 合并多头输出
        attn_output = attn_output.transpose(1, 2).reshape(B, 1, D)  # (B, 1, D)
        attn_output = self.out_proj(attn_output.squeeze(1))  # (B, D)

        # 残差连接 + 层归一化
        return self.norm(x + attn_output)


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=8):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


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

def normalize_adj(adj: sp.spmatrix) -> sp.coo_matrix:
    """
    对稀疏邻接矩阵做对称归一化：
    A_hat = A + I
    A_norm = D^{-1/2} A_hat D^{-1/2}
    返回 coo_matrix。
    """
    # 确保为 CSR，便于快速算术/求和
    adj = adj.tocsr()

    # 添加自环（如果已经有自环，sp.eye 会在对角上累加）
    adj_hat = adj + sp.eye(adj.shape[0], dtype=adj.dtype, format='csr')

    degrees = np.array(adj_hat.sum(axis=1)).flatten()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(degrees, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    # 构造对角矩阵并完成归一化（稀疏乘法）
    D_inv_sqrt = sp.diags(d_inv_sqrt, format='csr')
    adj_norm = D_inv_sqrt.dot(adj_hat).dot(D_inv_sqrt).tocoo()

    return adj_norm


def normalize_adj_torch(adj: torch.Tensor) -> torch.Tensor:
    """
    对 torch.sparse_coo_tensor 进行对称归一化（添加自环并执行 D^{-1/2} A D^{-1/2}）。
    输入和输出均为与输入同设备的稀疏张量。
    """
    device = adj.device
    N = adj.size(0)

    # 将稀疏张量转为 COO 格式以便操作
    adj = adj.coalesce()
    indices = adj.indices()
    values = adj.values()

    # 添加自环：构造对角线索引
    diag_indices = torch.arange(N, device=device).unsqueeze(0).repeat(2, 1)
    all_indices = torch.cat([indices, diag_indices], dim=1)
    all_values = torch.cat([values, torch.ones(N, device=device)], dim=0)

    # 构造新的稀疏矩阵 A_hat = A + I
    adj_hat = torch.sparse_coo_tensor(all_indices, all_values, size=(N, N), device=device).coalesce()

    # 计算度数（按列求和）
    deg = torch.sparse.sum(adj_hat, dim=1).to_dense()  # (N,)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # 构造 D^{-1/2}
    deg_inv_sqrt_mat = torch.diag(deg_inv_sqrt).to_sparse().to(device)

    # 执行归一化：D^{-1/2} @ A_hat @ D^{-1/2}
    # 注意 torch.spmm 不支持两个 sparse matmul，需手动实现或转换为 dense（小规模）
    # 此处采用逐元素乘法方式模拟归一化过程（推荐用于大规模稀疏图）

    # 获取归一化后的 values
    row, col = adj_hat.indices()[0], adj_hat.indices()[1]
    norm_values = adj_hat.values() * deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adj_norm = torch.sparse_coo_tensor(adj_hat.indices(), norm_values, size=(N, N), device=device)
    return adj_norm
