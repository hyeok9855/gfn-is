import torch


def fwd_tb(initial_state, gfn, log_reward_fn, exploration_std=0.0):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()

    delta = -(log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r)
    return delta, states, log_pfs, log_pbs, log_r


def bwd_tb(initial_state, gfn, log_reward_fn):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(initial_state, log_reward_fn)
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()

    delta = -(log_pfs.sum(-1) + log_fs[:, 0] - log_pbs.sum(-1) - log_r)
    return delta


def fwd_tb_avg(initial_state, gfn, log_reward_fn, exploration_std=0.0):
    states, log_pfs, log_pbs, _ = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()

    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    delta = -(log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1)))
    return delta, states, log_pfs, log_pbs, log_r


def bwd_tb_avg(initial_state, gfn, log_reward_fn):
    states, log_pfs, log_pbs, _ = gfn.get_trajectory_bwd(initial_state, log_reward_fn)
    with torch.no_grad():
        log_r = log_reward_fn(states[:, -1]).detach()

    log_Z = (log_r + log_pbs.sum(-1) - log_pfs.sum(-1)).mean(dim=0, keepdim=True)
    delta = -(log_Z + (log_pfs.sum(-1) - log_r - log_pbs.sum(-1)))
    return delta


def db(initial_state, gfn, log_reward_fn, exploration_std=0.0):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1]).detach()

    loss = 0.5 * ((log_pfs + log_fs[:, :-1] - log_pbs - log_fs[:, 1:]) ** 2).sum(-1)
    return loss, states, log_pfs, log_pbs, log_fs[:, -1]


def subtb(initial_state, gfn, log_reward_fn, coef_matrix, exploration_std=0.0):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn)
    with torch.no_grad():
        log_fs[:, -1] = log_reward_fn(states[:, -1]).detach()

    diff_logp = log_pfs - log_pbs
    diff_logp_padded = torch.cat(
        (torch.zeros((diff_logp.shape[0], 1)).to(diff_logp),
         diff_logp.cumsum(dim=-1)),
        dim=1)
    A1 = diff_logp_padded.unsqueeze(1) - diff_logp_padded.unsqueeze(2)
    A2 = log_fs[:, :, None] - log_fs[:, None, :] + A1
    A2 = A2 ** 2
    bs = states.shape[0]
    return torch.stack([torch.triu(A2[i] * coef_matrix, diagonal=1).sum() for i in range(A2.shape[0])]) * bs, states, log_pfs, log_pbs, log_fs[:, -1]
 

def bwd_mle(samples, gfn, log_reward_fn):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_bwd(samples, log_reward_fn)
    loss = -log_pfs.sum(-1)
    return loss


def pis(initial_state, gfn, log_reward_fn, exploration_std=0.0):
    states, log_pfs, log_pbs, log_fs = gfn.get_trajectory_fwd(initial_state, exploration_std, log_reward_fn, pis=True)
    with torch.enable_grad():
        log_r = log_reward_fn(states[:, -1])

    normalization_constant = float(1 / initial_state.shape[-1])
    loss = normalization_constant * (log_pfs.sum(-1) - log_pbs.sum(-1) - log_r)
    return loss
