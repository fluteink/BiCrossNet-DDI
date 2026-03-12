import logging
import os
import warnings
from argparse import Namespace
from typing import Tuple

from typing import Union, List
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score, accuracy_score
import numpy as np
import torch
from numpy.dtypes import Float64DType
from sklearn.metrics import roc_curve
import time

from model import DDI


class _ElapsedTimeFilter(logging.Filter):
    """Logging filter that injects an 'elapsed' attribute with seconds since start."""
    def __init__(self, start_time: float | None = None):
        super().__init__()
        self.start_time = start_time if start_time is not None else time.time()

    def filter(self, record: logging.LogRecord) -> bool:
        record.elapsed = time.time() - self.start_time
        return True


def save_checkpoint(path: str,
                    model,
                    args: Namespace = None):
    state = {
        'args': args,
        'state_dict': model.state_dict(),
    }
    torch.save(state, path)


def load_checkpoint(path: str,
                    current_args: Namespace = None,
                    cuda: bool = None,
                    logger: logging.Logger = None,
                    ddi: bool = False, model_type='final'):
    # Prefer a logger over print for any diagnostics
    logger = logger or logging.getLogger('train')
    debug = logger.debug

    # Check if file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Load model and args
    try:
        state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
    except Exception as e:
        # 第一次失败：尝试将常见的不可安全反序列化类加入 safe globals 后重试
        torch.serialization.add_safe_globals([
            Namespace,
            np._core.multiarray.scalar,  # 基础标量类型
            np.dtype,  # 通用dtype
            Float64DType,  # 具体的Float64DType
            np.float64  # 对应的数据类型（可能被引用）
        ])
        try:
            state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
        except Exception as e2:
            # 如果仍然失败，警告并回退到非 weights_only 加载（仅在信任文件来源时使用）
            warnings.warn(f"weights_only load failed ({e2}). Falling back to full load (may execute arbitrary code).")
            state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)

    args, loaded_state_dict = state['args'], state['state_dict']

    if current_args is not None:
        args = current_args

    args.cuda = cuda if cuda is not None else args.cuda
    # Create model and optimizer
    model = DDI(args)

    model_state_dict = model.state_dict()

    # Skip missing parameters and parameters of mismatched size
    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():

        if param_name not in model_state_dict:
            debug(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[param_name].shape != loaded_state_dict[param_name].shape:
            debug(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[param_name].shape}.')
        else:
            debug(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[param_name] = loaded_state_dict[param_name]

    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)

    if cuda:
        debug('Moving model to cuda')
        model = model.cuda()

    return model


def makedirs(path: str, isfile: bool = False):
    """
    Creates a directory given a path to either a directory or file.

    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    :param path: Path to a directory or file.
    :param isfile: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != '':
        os.makedirs(path, exist_ok=True)


def accuracy(targets: List[int], preds: List[float], threshold: float = 0.5) -> float:
    """
    Computes the accuracy of a binary prediction task using a given threshold for generating hard predictions.
    Alternatively, compute accuracy for a multiclass prediction task by picking the largest probability. 

    :param targets: A list of binary targets.
    :param preds: A list of prediction probabilities.
    :param threshold: The threshold above which a prediction is a 1 and below which (inclusive) a prediction is a 0
    :return: The computed accuracy.
    """
    if isinstance(preds[0], (list, tuple, np.ndarray)):  # multiclass
        hard_preds = [int(np.argmax(p)) for p in preds]
    else:
        hard_preds = [1 if p > threshold else 0 for p in preds]  # binary prediction
    return accuracy_score(targets, hard_preds)


def create_logger(name: str, save_dir: str = None, quiet: bool = False, run_id: str | None = None) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    - Standardized formatter with asctime and elapsed seconds from start.
    - Logs are written into a per-run directory (default: ./log/<name>/<run_id>/).

    :param name: The name of the logger.
    :param save_dir: The directory in which to save the logs. If None, use ./log/<name>/<run_id>/.
    :param quiet: Whether the stream handler should be quiet (i.e. print only important info).
    :param run_id: Optional run identifier for per-run separation.
    :return: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Avoid duplicate handlers if create_logger is called multiple times
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

    # Determine save directory
    run_id = run_id or time.strftime('%Y%m%d_%H%M%S')
    default_dir = os.path.join('log', name, run_id)
    save_dir = save_dir or default_dir
    makedirs(save_dir)

    # Elapsed time filter
    elapsed_filter = _ElapsedTimeFilter()

    # Formatter with timestamp and elapsed seconds
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)8s | %(name)s | +%(elapsed).1fs | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Stream handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO if quiet else logging.DEBUG)
    ch.addFilter(elapsed_filter)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handlers
    fh_v = logging.FileHandler(os.path.join(save_dir, 'verbose.log'), encoding='utf-8')
    fh_v.setLevel(logging.DEBUG)
    fh_v.addFilter(elapsed_filter)
    fh_v.setFormatter(formatter)

    fh_q = logging.FileHandler(os.path.join(save_dir, 'quiet.log'), encoding='utf-8')
    fh_q.setLevel(logging.INFO)
    fh_q.addFilter(elapsed_filter)
    fh_q.setFormatter(formatter)

    logger.addHandler(fh_v)
    logger.addHandler(fh_q)

    logger.debug(f'Logger initialized. save_dir={save_dir}, run_id={run_id}')
    return logger


def gen_preds(edges_pos, edges_neg, adj_rec):
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []
    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])

    return preds, preds_neg


def eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test):
    for i in range(int(0.5 * len(labels_all))):
        if preds_all[2 * i] > 0.95 and preds_all[2 * i + 1] > 0.95:
            preds_all[2 * i] = max(preds_all[2 * i], preds_all[2 * i + 1])
            preds_all[2 * i + 1] = preds_all[2 * i]
        else:
            preds_all[2 * i] = min(preds_all[2 * i], preds_all[2 * i + 1])
            preds_all[2 * i + 1] = preds_all[2 * i]
    # 检查是否有 NaN 值
    preds_all = np.nan_to_num(preds_all)
    fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    preds_all_ = []
    for p in preds_all:
        if p >= optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)
    return preds_all, preds_all_



def get_roc_score(
        rec,
        edges_pos: np.ndarray,
        edges_neg: Union[np.ndarray, List[list]],
        test=None) -> Tuple[float, float, float, float]:

    rec = rec.detach().cpu().numpy()
    adj_rec = rec

    preds, preds_neg = gen_preds(edges_pos, edges_neg, adj_rec)
    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    preds_all, preds_all_ = eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test)

    roc_score_v = roc_auc_score(labels_all, preds_all)
    ap_score_v = average_precision_score(labels_all, preds_all)
    f1_score_v = f1_score(labels_all, preds_all_)
    acc_score_v = accuracy_score(labels_all, preds_all_)

    # 仅在测试阶段记录混淆矩阵并使用日志记录
    if test:
        cm = confusion_matrix(labels_all, preds_all_)
        logger = logging.getLogger('train')
        logger.info('Confusion Matrix:\n%s', cm)

    return roc_score_v, ap_score_v, f1_score_v, acc_score_v

# def get_roc_score(
#         rec,
#         edges_pos: np.ndarray,
#         edges_neg: Union[np.ndarray, List[list]],
#         test=None) -> Tuple[float, float]:
#
#
#     rec = rec.detach().cpu().numpy()
#     adj_rec = rec
#
#     preds, preds_neg = gen_preds(edges_pos, edges_neg, adj_rec)
#     preds_all = np.hstack([preds, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
#     preds_all, preds_all_ = eval_threshold(labels_all, preds_all, preds, edges_pos, edges_neg, adj_rec, test)
#
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)
#     f1_score_ = f1_score(labels_all, preds_all_)
#     acc_score = accuracy_score(labels_all, preds_all_)
#     return roc_score, ap_score, f1_score_, acc_score





def get_roc_score2(preds, edges_pos, edges_neg, threshold=0.5, test=False):
    preds = torch.sigmoid(preds)  # 将输出转换为概率
    preds = preds.cpu().detach().numpy()

    # 处理正样本
    preds_pos = []
    for e in edges_pos:
        preds_pos.append(preds[e[0], e[1]])

    # 处理负样本
    preds_neg = []
    for e in edges_neg:
        preds_neg.append(preds[e[0], e[1]])

    # 合并所有预测结果
    preds_pos = np.asarray(preds_pos, dtype=float)
    preds_neg = np.asarray(preds_neg, dtype=float)
    y_true = [1] * len(preds_pos) + [0] * len(preds_neg)
    y_scores = np.concatenate([preds_pos, preds_neg])
    y_pred = (y_scores >= threshold).astype(int)

    # 计算指标
    roc_score_v = roc_auc_score(y_true, y_scores)
    ap_score_v = average_precision_score(y_true, y_scores)
    f1_score_val = f1_score(y_true, y_pred)
    acc_score = accuracy_score(y_true, y_pred)

    if test:
        return roc_score_v, ap_score_v, f1_score_val, acc_score, y_true, y_pred

    return roc_score_v, ap_score_v, f1_score_val, acc_score
