import torch

from models import GFN


def tb_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_Z: torch.Tensor,
    log_r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    tb_discrepancy = log_r + log_pbs.sum(-1) - log_pfs.sum(-1) - log_Z
    return tb_discrepancy.detach(), 0.5 * (tb_discrepancy ** 2)


def tb_avg_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_r: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    tb_avg_discrepancy = log_r + log_pbs.sum(-1) - log_pfs.sum(-1) - log_Z
    return tb_avg_discrepancy.detach(), 0.5 * (tb_avg_discrepancy ** 2)


def db_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    db_discrepancy = log_fs[:, 1:] + log_pbs - log_pfs - log_fs[:, :-1]
    return db_discrepancy.detach(), 0.5 * (db_discrepancy ** 2)


def subtb_loss(
    log_pfs: torch.Tensor,
    log_pbs: torch.Tensor,
    log_fs: torch.Tensor,
    coef_matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    diff_logp = log_pfs - log_pbs
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp), diff_logp.cumsum(dim=-1)),
        dim=1,
    )
    A = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
    subtb_discrepancy = log_fs[:, :, None] - log_fs[:, None, :] + A
    subtb_discrepancy_squared = subtb_discrepancy ** 2
    subtb_loss = log_fs.shape[0] * torch.stack(
        [
            torch.triu(subtb_discrepancy_squared[i] * coef_matrix, diagonal=1).sum()
            for i in range(subtb_discrepancy_squared.shape[0])
        ]
    )
    return subtb_discrepancy.detach(), subtb_loss


def bwd_mle(samples, gfn: GFN, log_reward_fn):
    _, log_pfs, _, _ = gfn.get_trajectory_bwd(samples, log_reward_fn)
    loss = -log_pfs.sum(-1)
    return loss
