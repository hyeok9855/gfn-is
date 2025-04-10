import torch


def tb_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_Z: torch.Tensor,
    log_r: torch.Tensor,
) -> torch.Tensor:
    tb_discrepancy = log_Z + log_pfs.sum(-1) - log_r - log_pbs.sum(-1)
    return tb_discrepancy ** 2


def tb_avg_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_r: torch.Tensor,
) -> torch.Tensor:
    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    tb_avg_discrepancy = log_r + log_pbs.sum(-1) - log_pfs.sum(-1) - log_Z
    return tb_avg_discrepancy ** 2


def db_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
) -> torch.Tensor:
    db_discrepancy = log_fs[:, 1:] + log_pbs - log_pfs - log_fs[:, :-1]
    return (db_discrepancy ** 2).sum(-1)


def subtb_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    coef_matrix: torch.Tensor,
) -> torch.Tensor:
    diff_logp = log_pfs - log_pbs  # (bs, T)
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp), diff_logp.cumsum(dim=-1)),
        dim=1,
    )  # (bs, T+1)
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)  # (bs, T+1, T+1)
    A2 = log_fs[:, :, None] - log_fs[:, None, :] + A1  # (bs, T+1, T+1)
    subtb_losses = torch.triu(
        (A2 ** 2) * coef_matrix.unsqueeze(0), diagonal=1
    ).sum((1, 2))
    return subtb_losses


def get_gfn_loss(
    loss_type: str,
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    subtb_coef_matrix: torch.Tensor | None = None,
    ndim: int | None = None,
) -> torch.Tensor:
    if loss_type == 'tb':
        losses = tb_loss(log_pfs, log_pbs, log_fs[:, 0], log_fs[:, -1])
    elif loss_type == 'tb-avg':
        losses = tb_avg_loss(log_pfs, log_pbs, log_fs[:, -1])
    elif loss_type == 'db':
        losses = db_loss(log_pfs, log_pbs, log_fs)
    elif loss_type == 'subtb':
        assert subtb_coef_matrix is not None
        losses = subtb_loss(log_pfs, log_pbs, log_fs, subtb_coef_matrix)
    elif loss_type == 'pis':
        assert ndim is not None
        losses = (1 / ndim) * (log_pfs.sum(-1) - log_pbs.sum(-1) - log_fs[:, -1])
    else:
        raise ValueError(f'Invalid training loss: {loss_type}')

    return losses
