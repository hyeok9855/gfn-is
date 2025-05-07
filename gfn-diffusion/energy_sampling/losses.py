import numpy as np
import torch


def tb_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_Z: torch.Tensor,
    log_r: torch.Tensor,
) -> torch.Tensor:
    tb_discrepancy = log_Z + log_pfs.sum(-1) - log_r - log_pbs.sum(-1)
    return tb_discrepancy**2


def db_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
) -> torch.Tensor:
    db_discrepancy = log_fs[:, :-1] + log_pfs - log_fs[:, 1:] - log_pbs
    return (db_discrepancy**2).mean(-1)


def subtb_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    coef_matrix: torch.Tensor,  # (T+1, T+1)
) -> torch.Tensor:
    diff_logp = log_pfs - log_pbs  # (bs, T)
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp), diff_logp.cumsum(dim=-1)),
        dim=1,
    )  # (bs, T+1)
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)  # (bs, T+1, T+1)
    A2 = log_fs.unsqueeze(2) - log_fs.unsqueeze(1) + A1  # (bs, T+1, T+1)
    A2 = torch.triu(A2, diagonal=1) ** 2
    subtb_losses = (A2 * coef_matrix.unsqueeze(0)).sum((1, 2))
    return subtb_losses


def subtb_chunk_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    subtb_chunk_size: int,
) -> torch.Tensor:
    db_discrepancy = log_fs[:, :-1] + log_pfs - log_fs[:, 1:] - log_pbs
    # (bs, T)
    bs, T = db_discrepancy.shape
    assert T % subtb_chunk_size == 0

    db_discrepancy_chunked = db_discrepancy.reshape(bs, -1, subtb_chunk_size)
    # (bs, T/L, L)
    subtb_chunk_losses = db_discrepancy_chunked.sum(dim=-1)
    # (bs, T/L)
    return (subtb_chunk_losses**2).mean(-1)


def logvar_loss(
    log_pfs: torch.Tensor,  # (bs, T)
    log_pbs: torch.Tensor,  # (bs, T)
    log_r: torch.Tensor,  # (bs,)
) -> torch.Tensor:
    rnd = log_r + log_pbs.sum(-1) - log_pfs.sum(-1)  # (bs,)
    return (rnd - rnd.mean(dim=0, keepdim=True)) ** 2  # (bs,)


def get_loss(
    loss_type: str,
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    subtb_coef_matrix: torch.Tensor | None = None,
    subtb_chunk_size: int = 0,
    ndim: int | None = None,
) -> torch.Tensor:
    if loss_type == "tb":
        losses = tb_loss(log_pfs, log_pbs, log_fs[:, 0], log_fs[:, -1])
    elif loss_type == "logvar":
        losses = logvar_loss(log_pfs, log_pbs, log_fs[:, -1])
    elif loss_type == "db":
        losses = db_loss(log_pfs, log_pbs, log_fs)
    elif loss_type == "subtb":
        if subtb_chunk_size > 0:  # Chunk-based subtb
            losses = subtb_chunk_loss(log_pfs, log_pbs, log_fs, subtb_chunk_size)
        else:
            assert subtb_coef_matrix is not None
            losses = subtb_loss(log_pfs, log_pbs, log_fs, subtb_coef_matrix)
    elif loss_type == "pis":
        assert ndim is not None
        losses = (1 / ndim) * (log_pfs.sum(-1) - log_pbs.sum(-1) - log_fs[:, -1])
    else:
        raise ValueError(f"Invalid training loss: {loss_type}")

    return losses


def cal_subtb_coef_matrix(lamda: float, T: int) -> torch.Tensor:
    """
    diff_matrix: (T+1, T+1)
     0,  1,  2, ...,   T
    -1,  0,  1, ..., T-1
    -2, -1,  0, .... T-2
    ...

    self.coef[i, j] = lamda^(j-i) / total_lambda  if i < j else 0.
    """
    assert lamda >= 0
    if lamda == 0:  # DB
        ones = torch.ones(T + 1, T + 1)
        coef = torch.triu(ones, diagonal=1) - torch.triu(ones, diagonal=2)
        coef = coef / T
    elif lamda == float("inf"):  # TB if lambda is inf
        coef = torch.zeros(T + 1, T + 1)
        coef[0, -1] = 1.0
    else:
        range_vals = torch.arange(T + 1)
        diff_matrix = range_vals - range_vals.view(-1, 1)
        B = np.log(lamda) * diff_matrix
        B[diff_matrix <= 0] = -np.inf
        log_total_lambda = torch.logsumexp(B.view(-1), dim=0)
        coef = torch.exp(B - log_total_lambda)
    return coef
