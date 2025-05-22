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
    n_chunks: int,
) -> torch.Tensor:
    db_discrepancy = log_fs[:, :-1] + log_pfs - log_fs[:, 1:] - log_pbs
    # (bs, T)
    bs, T = db_discrepancy.shape
    assert T % n_chunks == 0

    db_discrepancy_chunked = db_discrepancy.reshape(bs, n_chunks, -1)
    # (bs, n_chunks, T/n_chunks)
    subtb_chunk_losses = db_discrepancy_chunked.sum(dim=-1)
    # (bs, n_chunks)
    return (subtb_chunk_losses**2).mean(-1)


def logvar_loss(
    log_pfs: torch.Tensor,  # (bs, T)
    log_pbs: torch.Tensor,  # (bs, T)
    log_r: torch.Tensor,  # (bs,)
) -> torch.Tensor:
    rnd = log_r + log_pbs.sum(-1) - log_pfs.sum(-1)  # (bs,)
    return (rnd - rnd.mean(dim=0, keepdim=True)) ** 2  # (bs,)


def sublogvar_loss(
    log_pfs: torch.Tensor,  # (bs//sublogvar_K, sublogvar_K, T)
    log_pbs: torch.Tensor,  # (bs//sublogvar_K, sublogvar_K, T)
    log_r: torch.Tensor | None = None,  # (bs//sublogvar_K, sublogvar_K) or None
) -> torch.Tensor:
    rnd = log_pbs.sum(-1) - log_pfs.sum(-1)  # (bs//sublogvar_K, sublogvar_K)
    if log_r is not None:  # None for bwd sub-trajectories
        rnd = log_r + rnd
    return (rnd - rnd.mean(dim=1, keepdim=True)) ** 2  # (bs//sublogvar_K, sublogvar_K)


def get_loss(
    loss_type: str,
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    invtemp: float = 1.0,
    subtb_coef_matrix: torch.Tensor | None = None,
    subtb_n_chunks: int = 0,
    ndim: int | None = None,
    sublogvar_K: int = 1,
    ts: torch.Tensor | None = None,
    curr_t: torch.Tensor | None = None,
) -> torch.Tensor:
    # Apply inverse temperature to the log reward
    log_fs[:, -1] = log_fs[:, -1] * invtemp

    if loss_type == "tb":
        losses = tb_loss(log_pfs, log_pbs, log_fs[:, 0], log_fs[:, -1])
    elif loss_type == "db":
        losses = db_loss(log_pfs, log_pbs, log_fs)
    elif loss_type == "subtb":
        if subtb_n_chunks > 0:  # Chunk-based subtb
            losses = subtb_chunk_loss(log_pfs, log_pbs, log_fs, subtb_n_chunks)
        else:
            assert subtb_coef_matrix is not None
            losses = subtb_loss(log_pfs, log_pbs, log_fs, subtb_coef_matrix)
    elif loss_type == "pis":
        assert ndim is not None
        losses = (1 / ndim) * (log_pfs.sum(-1) - log_pbs.sum(-1) - log_fs[:, -1])
    elif loss_type == "logvar":
        if sublogvar_K == 1:
            losses = logvar_loss(log_pfs, log_pbs, log_fs[:, -1])
        else:  # subtrajectory-based logvar
            assert ts is not None and curr_t is not None
            curr_t_idx = torch.where(ts == curr_t.unsqueeze(1))[1]  # (bs,)

            bs, T = log_pfs.shape
            arange = torch.arange(bs).unsqueeze(1)
            dummy = torch.zeros(bs, 1).to(log_pfs)
            log_pfs = torch.cat([log_pfs, dummy], dim=1)  # idx T is dummy
            log_pbs = torch.cat([log_pbs, dummy], dim=1)  # idx T is dummy

            t_idx_fwdtraj = torch.arange(T).to(curr_t_idx).repeat(bs, 1)
            t_idx_fwdtraj = (t_idx_fwdtraj + curr_t_idx.unsqueeze(1)).clamp(min=0, max=T)
            log_pfs_fwdtraj = log_pfs[arange, t_idx_fwdtraj].reshape(
                bs // sublogvar_K, sublogvar_K, -1
            )
            log_pbs_fwdtraj = log_pbs[arange, t_idx_fwdtraj].reshape(
                bs // sublogvar_K, sublogvar_K, -1
            )
            log_r_fwdtraj = log_fs[arange, -1].reshape(bs // sublogvar_K, sublogvar_K)
            losses_fwdtraj = sublogvar_loss(log_pfs_fwdtraj, log_pbs_fwdtraj, log_r_fwdtraj)

            t_idx_bwdtraj = torch.arange(T).to(curr_t_idx).repeat(bs, 1)
            t_idx_bwdtraj = torch.where(t_idx_bwdtraj >= curr_t_idx.unsqueeze(1), T, t_idx_bwdtraj)
            log_pfs_bwdtraj = log_pfs[arange, t_idx_bwdtraj].reshape(
                bs // sublogvar_K, sublogvar_K, -1
            )
            log_pbs_bwdtraj = log_pbs[arange, t_idx_bwdtraj].reshape(
                bs // sublogvar_K, sublogvar_K, -1
            )
            losses_bwdtraj = sublogvar_loss(log_pfs_bwdtraj, log_pbs_bwdtraj)

            losses = (losses_fwdtraj + losses_bwdtraj).flatten()
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
