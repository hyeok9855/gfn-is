import numpy as np
import torch

from utils.misc_utils import huber_loss


def tb_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_Z: torch.Tensor,
    log_r: torch.Tensor,
    quantile: float = 1.0,
) -> torch.Tensor:
    tb_discrepancy = log_Z + log_pfs.sum(-1) - log_r - log_pbs.sum(-1)
    return huber_loss(tb_discrepancy, quantile=quantile)


def tb_avg_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_r: torch.Tensor,
    quantile: float = 1.0,
) -> torch.Tensor:
    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    tb_avg_discrepancy = log_r + log_pbs.sum(-1) - log_pfs.sum(-1) - log_Z
    return huber_loss(tb_avg_discrepancy, quantile=quantile)


def db_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    quantile: float = 1.0,
) -> torch.Tensor:
    db_discrepancy = log_fs[:, 1:] + log_pbs - log_pfs - log_fs[:, :-1]
    return huber_loss(db_discrepancy, quantile=quantile).sum(-1)


def subtb_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    coef_matrix: torch.Tensor,  # (T+1, T+1)
    quantile: float = 1.0,
) -> torch.Tensor:
    diff_logp = log_pfs - log_pbs  # (bs, T)
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp), diff_logp.cumsum(dim=-1)),
        dim=1,
    )  # (bs, T+1)
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)  # (bs, T+1, T+1)
    A2 = log_fs[:, :, None] - log_fs[:, None, :] + A1  # (bs, T+1, T+1)
    A2 = huber_loss(torch.triu(A2, diagonal=1), quantile=quantile)
    subtb_losses = (A2 * coef_matrix.unsqueeze(0)).sum((1, 2))
    return subtb_losses


def get_gfn_loss(
    loss_type: str,
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    subtb_coef_matrix: torch.Tensor | None = None,
    ndim: int | None = None,
    huber_quantile: float = 1.0,
) -> torch.Tensor:
    if loss_type == "tb":
        losses = tb_loss(log_pfs, log_pbs, log_fs[:, 0], log_fs[:, -1], quantile=huber_quantile)
    elif loss_type == "tb-avg":
        losses = tb_avg_loss(log_pfs, log_pbs, log_fs[:, -1], quantile=huber_quantile)
    elif loss_type == "db":
        losses = db_loss(log_pfs, log_pbs, log_fs, quantile=huber_quantile)
    elif loss_type == "subtb":
        assert subtb_coef_matrix is not None
        losses = subtb_loss(log_pfs, log_pbs, log_fs, subtb_coef_matrix, quantile=huber_quantile)
    elif loss_type == "pis":
        assert ndim is not None
        losses = (1 / ndim) * (log_pfs.sum(-1) - log_pbs.sum(-1) - log_fs[:, -1])
    else:
        raise ValueError(f"Invalid training loss: {loss_type}")

    return losses


def cal_subtb_coef_matrix(lamda: float, N: int) -> torch.Tensor:
    """
    diff_matrix: (N+1, N+1)
     0,  1,  2, ...,   N
    -1,  0,  1, ..., N-1
    -2, -1,  0, .... N-2
    ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    assert lamda >= 0
    if lamda == 0:  # DB
        ones = torch.ones(N + 1, N + 1)
        coef = torch.triu(ones, diagonal=1) - torch.triu(ones, diagonal=2)
        coef = coef / N
    elif lamda == float("inf"):  # TB if lambda is inf
        coef = torch.zeros(N + 1, N + 1)
        coef[0, -1] = 1.0
    else:
        range_vals = torch.arange(N + 1)
        diff_matrix = range_vals - range_vals.view(-1, 1)
        B = np.log(lamda) * diff_matrix
        B[diff_matrix <= 0] = -np.inf
        log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
        coef = torch.exp(B - log_total_lambda)
    return coef
