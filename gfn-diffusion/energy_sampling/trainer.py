from typing import Callable, Literal
import warnings

import torch

from buffers import BaseBuffer, GISTerminalStateBuffer, IntermediateStateBuffer, TerminalStateBuffer
from energies import BaseEnergy, IntermediateEnergy
from losses import cal_subtb_coef_matrix, get_loss
from mcmcs import BaseMCMC
from models import GFN
from utils.eval_utils import density_metrics, distribution_distance_metrics
from utils.misc_utils import linear_annealing
from utils.plot_utils import visualize
from utils.sampling_utils import get_sampling_func
from utils.train_utils import binary_search_smoothing


class Trainer:
    def __init__(
        self,
        energy: BaseEnergy,
        gfn_model: GFN,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.MultiStepLR | None,
        clip_grad_norm: float,
        loss_type: Literal["tb", "db", "subtb", "logvar", "pis", "mle"],
        subtb_lambda: float,
        subtb_n_chunks: int,
        sublogvar_K: int,
        n_epochs: int,
        bwd_to_fwd_ratio: float,
        buffer: BaseBuffer | None,
        buffer_save_interval: int,
        prefill_epochs: int,
        batch_size: int,
        train_discretizer: Callable[[int, int], torch.Tensor],
        train_T: int,
        weighting: bool,
        resampling: bool,
        resampling_strategy: Literal["multinomial", "stratified", "systematic"],
        alternating: bool,
        target_ess: float,
        smoothing_strategy: Literal["temper", "clip_above", "clip_below"],
        mcmc: BaseMCMC | None,
        mcmc_freq: int,
        mcmc_batch_size: int,
        epsilon: float,
        anneal_epsilon: bool,
        invtemp: float,
        invtemp_anneal: bool,
        eval_batch_size: int,
        eval_discretizer: Callable[[int, int], torch.Tensor],
        eval_T: int,
        eval_weighting: bool,
        eval_resampling: bool,
        plot_gt: bool,
        plot_t_idx: list[int],
        plot_buffer_t_idx: list[int],
    ):

        self.energy = energy
        self.gfn_model = gfn_model
        self.device = gfn_model.device

        # Optimizer and scheduler
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm

        # Loss
        self.loss_type = loss_type
        self.subtb_n_chunks = subtb_n_chunks
        self.subtb_coef_matrix = None
        if loss_type == "subtb" and subtb_n_chunks == 0:  # chunk-based subtb
            self.subtb_coef_matrix = cal_subtb_coef_matrix(subtb_lambda, train_T).to(self.device)
        self.sublogvar_K = sublogvar_K

        # Training parameters and buffer
        self.n_epochs = n_epochs
        if bwd_to_fwd_ratio < 1:
            self.bwd_to_fwd_ratio = None
            self.fwd_to_bwd_ratio = round(1 / bwd_to_fwd_ratio)
        else:
            self.bwd_to_fwd_ratio = int(bwd_to_fwd_ratio)
            self.fwd_to_bwd_ratio = None
        self.buffer = buffer
        self.buffer_save_interval = buffer_save_interval
        self.prefill_epochs = prefill_epochs if self.buffer is not None else 0

        # Sampling parameters
        self.batch_size = batch_size
        self.train_discretizer = train_discretizer
        self.train_T = train_T

        # Weighted or resampled training
        self._weighting = weighting
        self._resampling = resampling
        self.resampling_strategy = resampling_strategy
        self.alternating = alternating
        self._alternating_flag = True

        # Importance sampling
        if (
            (self.buffer is not None and self.buffer.prioritization == "normalized_iw")
            or weighting
            or resampling
        ):
            self.target_ess = target_ess
        else:  # otherwise, we don't need smoothing; disable it by setting target_ess to 0.0
            self.target_ess = 0.0
        self.smoothing_strategy = smoothing_strategy

        # MCMC
        self.mcmc = mcmc
        self.mcmc_freq = mcmc_freq
        self.mcmc_batch_size = mcmc_batch_size
        # Misc
        self.epsilon = epsilon
        self.anneal_epsilon = anneal_epsilon
        self.invtemp = invtemp
        self.invtemp_anneal = invtemp_anneal

        # Eval and Plot
        self.eval_batch_size = eval_batch_size
        self.eval_discretizer = eval_discretizer
        self.eval_T = eval_T
        self.eval_weighting = eval_weighting
        self.eval_resampling = eval_resampling
        self.plot_gt = plot_gt
        self.plot_t_idx = plot_t_idx
        self.plot_buffer_t_idx = plot_buffer_t_idx

    @property
    def weighting(self) -> bool:
        return self._weighting and self._alternating_flag

    @property
    def resampling(self) -> bool:
        return self._resampling and self._alternating_flag

    def train_step(self, it: int) -> float:
        self.gfn_model.train()
        self.energy.invtemp = (
            self.invtemp
            if not self.invtemp_anneal
            else linear_annealing(it, int(0.8 * self.n_epochs), self.invtemp, 1.0, descending=False)
        )

        if it < self.prefill_epochs:
            # Prefill buffer with forward sampling
            assert self.buffer is not None
            loss = self.fwd_train_step(it)
        elif it == self.prefill_epochs:
            # We initialize the flow_model with unbiased estimator of the log-partition function
            # TODO, after implementing GIS-like buffer
            pass

        if self.loss_type == "mle":
            loss = self.bwd_train_step(it)
        elif self.buffer is None:
            loss = self.fwd_train_step(it)
        else:  # self.buffer is not None
            if (
                (self.bwd_to_fwd_ratio is not None and it % (self.bwd_to_fwd_ratio + 1) == 0)
                or (self.fwd_to_bwd_ratio is not None and it % (self.fwd_to_bwd_ratio + 1) != 0)
            ) or it < self.prefill_epochs:
                loss = self.fwd_train_step(it)
            else:
                loss = self.bwd_train_step(it)

        # MCMC buffer augmentation
        if self.mcmc is not None and (it > 0 and (it - self.prefill_epochs) % self.mcmc_freq == 0):
            # TODO: support for intermediate states
            assert isinstance(self.buffer, TerminalStateBuffer)
            assert self.buffer.prioritization in ["normalized_iw", "iw", "none", "target"]

            buf_xs, _, indices = self.buffer.sample(self.mcmc_batch_size)

            # Augment the buffer with samples from MCMC
            mcmc_xs, mcmc_log_rs = self.mcmc.sample(buf_xs)

            data_dict = {
                "states": mcmc_xs.reshape(-1, self.energy.ndim),
                "log_fs": mcmc_log_rs.reshape(-1),
            }

            if isinstance(self.buffer, GISTerminalStateBuffer):
                log_iws = self.buffer.log_iws_dataset[indices]
                log_iws = log_iws.unsqueeze(0).repeat(mcmc_log_rs.shape[0], 1)
                data_dict["log_iws"] = log_iws.reshape(-1)
            elif self.buffer.normalized_iws_dataset is not None:
                normalized_iws = self.buffer.normalized_iws_dataset[indices]
                normalized_iws = normalized_iws.unsqueeze(0).repeat(mcmc_log_rs.shape[0], 1)
                data_dict["normalized_iws"] = normalized_iws.reshape(-1)

            self.buffer.add(**data_dict)

        if loss.isnan():
            raise ValueError(f"Loss is NaN")

        if it < self.prefill_epochs or loss.isinf() or loss > 1e28:
            return loss.item()

        loss.backward()
        if self.clip_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.gfn_model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.gfn_model.zero_grad()

        return loss.item()

    def fwd_train_step(self, it: int) -> torch.Tensor:
        init_states = torch.zeros(self.batch_size, self.energy.ndim).to(self.device)
        ts = self.train_discretizer(self.batch_size, self.train_T).to(self.device)
        epsilon = (
            self.epsilon
            if not self.anneal_epsilon
            else linear_annealing(it, int(self.n_epochs * 0.8), 0.0, self.epsilon, descending=True)
        )

        # Forward sampling
        states, log_pfs, log_pbs, log_fs, log_pfs_exp = self.gfn_model.get_trajectory_fwd(
            init_states, ts, epsilon=epsilon, pis=self.loss_type == "pis"
        )

        # Compute losses
        losses = get_loss(
            self.loss_type,
            log_pfs,
            log_pbs,
            log_fs,
            subtb_coef_matrix=self.subtb_coef_matrix,
            subtb_n_chunks=self.subtb_n_chunks,
            ndim=self.energy.ndim,
        )

        normalized_iws_0t = log_iws_0t = None
        if (
            (self.buffer is not None and self.buffer.prioritization in ["iw", "normalized_iw"])
            or self.weighting
            or self.resampling
        ):
            log_iws_0t = (log_fs[:, 1:] + log_pbs.cumsum(-1) - log_pfs_exp.cumsum(-1)).detach()
            if self.target_ess != 0.0:
                _target_ess = (
                    self.target_ess * self.batch_size
                    if 0.0 <= self.target_ess <= 1.0
                    else self.target_ess
                )
                assert 1.0 < _target_ess <= self.batch_size, f"Invalid target ESS: {_target_ess}"

                log_iws_0t_smoothed = binary_search_smoothing(
                    log_iws_0t, _target_ess, self.smoothing_strategy
                )
                normalized_iws_0t = log_iws_0t_smoothed.softmax(dim=0)  # (bs, T)
            else:
                normalized_iws_0t = log_iws_0t.softmax(dim=0)  # (bs, T)

        # Add data to buffer
        if self.buffer is not None:
            data_dict = {"states": states[:, 1:], "log_fs": log_fs[:, 1:]}  # (bs, T) both
            if self.buffer.prioritization == "loss":
                assert isinstance(self.buffer, TerminalStateBuffer)
                data_dict["losses"] = losses.unsqueeze(1)  # (bs, 1)
            elif self.buffer.prioritization == "iw":
                assert log_iws_0t is not None
                data_dict["log_iws"] = log_iws_0t  # (bs, T)
            elif self.buffer.prioritization == "normalized_iw":
                assert normalized_iws_0t is not None
                data_dict["normalized_iws"] = normalized_iws_0t  # (bs, T)

            if isinstance(self.buffer, TerminalStateBuffer):
                data_dict = {k: v[:, -1] for k, v in data_dict.items()}
            else:  # IntermediateStateBuffer
                data_dict["ts"] = ts[:, 1:]  # (bs, T)
                if self.buffer_save_interval > 0:
                    data_dict = {
                        k: v[:, self.buffer_save_interval - 1 :: self.buffer_save_interval]
                        for k, v in data_dict.items()
                    }
            self.buffer.add(**data_dict)

        # Weighting or resampling here is supported only on the trajectory level
        # TODO: support for the transition-level weighting or resampling
        if self.weighting or self.resampling:
            assert normalized_iws_0t is not None
            x_iws = normalized_iws_0t[:, -1]
            if self.weighting:
                loss = (x_iws * losses).sum()
            else:  # resampling
                indices = get_sampling_func(self.resampling_strategy)(  # type: ignore
                    x_iws, self.batch_size, True
                )
                loss = losses[indices].mean()
        else:
            loss = losses.mean()

        if self.alternating:
            self._alternating_flag = not self._alternating_flag

        return loss

    def bwd_train_step(self, it: int) -> torch.Tensor:
        sub_logvar_params = {}

        if self.loss_type == "mle":
            gt_xs, gt_log_rewards = self.energy.cached_sample(self.batch_size, seed=it)
            ts = self.train_discretizer(self.batch_size, self.train_T).to(self.device)
            _, log_pfs, log_pbs, log_fs = self.gfn_model.get_trajectory_bwd(
                gt_xs, ts, gt_log_rewards
            )
            # mle over trajectories
            loss = -log_pfs.sum(-1).mean()

        else:  # self.loss_type != "mle"
            assert self.buffer is not None
            if isinstance(self.buffer, TerminalStateBuffer):
                buf_xs, buf_log_rs, indices = self.buffer.sample(self.batch_size)
                # each with shape (bs,)

                # Construct complete trajectories
                ts = self.train_discretizer(self.batch_size, self.train_T).to(self.device)
                _, log_pfs, log_pbs, log_fs = self.gfn_model.get_trajectory_bwd(
                    buf_xs, ts, buf_log_rs
                )

            elif isinstance(self.buffer, IntermediateStateBuffer):
                # # Option 1: Construct transitions / subtrajectories (chunks)
                # chunk_size = T // subtb_n_chunks  # assumption: T is divisible by subtb_n_chunks
                # buf_states, buf_ts, buf_log_fs, indices = buffer.sample(batch_size * subtb_n_chunks)
                # # each with shape (bs,)

                # # TODO: support for other discretizers
                # assert discretizer.__name__ == "uniform_discretizer"
                # sub_ts = discretizer(batch_size * subtb_n_chunks, chunk_size).to(device) / subtb_n_chunks

                # # clamp to avoid negative time steps from floating point error
                # ts = (buf_ts.unsqueeze(-1) + sub_ts - sub_ts[:, [-1]]).clamp(min=0.0)
                # _, log_pfs, log_pbs, log_fs = gfn_model.get_subtrajectory_bwd(
                #     buf_states, ts, buf_log_fs, energy.log_reward
                # )

                # Option 2: Construct complete trajectories by sampling both backward and forward
                if self.loss_type == "logvar" and self.sublogvar_K > 1:
                    assert self.batch_size % self.sublogvar_K == 0
                    buf_states, buf_ts, _, indices = self.buffer.sample(
                        self.batch_size // self.sublogvar_K
                    )
                    buf_states = (
                        buf_states.unsqueeze(1).repeat(1, self.sublogvar_K, 1).flatten(0, 1)
                    )
                    buf_ts = buf_ts.unsqueeze(1).repeat(1, self.sublogvar_K).flatten(0, 1)
                else:
                    buf_states, buf_ts, _, indices = self.buffer.sample(self.batch_size)

                # TODO: support for other discretizers
                try:
                    assert self.train_discretizer.__name__ == "uniform_discretizer"
                except:  # for partial function
                    assert self.train_discretizer.func.__name__ == "uniform_discretizer"
                ts = self.train_discretizer(self.batch_size, self.train_T).to(self.device)

                _, log_pfs, log_pbs, log_fs, _ = self.gfn_model.get_trajectory_fwd_and_bwd(
                    buf_states, ts, buf_ts, epsilon=0.0  # TODO: support for epsilon
                )
                sub_logvar_params = {
                    "sublogvar_K": self.sublogvar_K,
                    "ts": ts,
                    "curr_t": buf_ts,
                }

            else:
                raise ValueError(f"Invalid buffer type: {type(self.buffer)}")

            losses = get_loss(
                self.loss_type,
                log_pfs,
                log_pbs,
                log_fs,
                subtb_coef_matrix=self.subtb_coef_matrix,
                subtb_n_chunks=self.subtb_n_chunks,
                ndim=self.energy.ndim,
                **sub_logvar_params,  # subtrajectory-based logvar specific
            )

            if self.buffer.prioritization == "loss":
                self.buffer.update(indices, losses=losses)

            loss = losses.mean()

        return loss

    def eval_and_plot(
        self,
        data_size: int,
        full_eval: bool,
        final_eval: bool = False,
        plot: bool = False,
    ) -> dict:
        metrics, model_trajs, weights, sample_xs_r, buffer_xs = self.eval_step(
            data_size, full_eval, final_eval
        )
        if plot:
            images = self.plot_step(model_trajs, weights, sample_xs_r, buffer_xs, self.plot_gt)
            metrics.update(images)
            self.plot_gt = False  # disable plotting gt after first plot
        return metrics

    @torch.no_grad()
    def eval_step(
        self,
        data_size: int,
        full_eval: bool = False,
        final_eval: bool = False,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        self.gfn_model.eval()

        metrics = {}

        eval_batch_size = min(self.eval_batch_size, data_size)

        with torch.no_grad():
            divisible = data_size % eval_batch_size == 0
            n_epochs = data_size // eval_batch_size + (1 if not divisible else 0)

            model_trajs, log_pfs, log_pbs, log_fs, log_rewards = [], [], [], [], []

            for i in range(n_epochs):
                init_state = torch.zeros(eval_batch_size, self.energy.ndim).to(
                    self.gfn_model.device
                )
                ts = self.eval_discretizer(eval_batch_size, self.eval_T).to(self.gfn_model.device)
                _model_trajs, _log_pfs, _log_pbs, _log_fs, _ = self.gfn_model.get_trajectory_fwd(
                    init_state, ts, epsilon=0.0, pis=self.loss_type == "pis"
                )
                _log_rewards = self.energy.log_reward(_model_trajs[:, -1], temper=False)
                model_trajs.append(_model_trajs)
                log_pfs.append(_log_pfs)
                log_pbs.append(_log_pbs)
                log_fs.append(_log_fs)
                log_rewards.append(_log_rewards)
            model_trajs = torch.cat(model_trajs, dim=0)[:data_size]
            sample_xs = model_trajs[:, -1]
            log_pfs = torch.cat(log_pfs, dim=0)[:data_size]
            log_pbs = torch.cat(log_pbs, dim=0)[:data_size]
            log_fs = torch.cat(log_fs, dim=0)[:data_size]
            log_rewards = torch.cat(log_rewards, dim=0)[:data_size]

            try:
                gt_xs, gt_log_rewards = self.energy.cached_sample(data_size)
                gt_log_pfs, gt_log_pbs = [], []
                for i in range(n_epochs):
                    gt_xs_batch = gt_xs[i * eval_batch_size : (i + 1) * eval_batch_size]
                    gt_log_rewards_batch = gt_log_rewards[
                        i * eval_batch_size : (i + 1) * eval_batch_size
                    ]
                    ts = self.eval_discretizer(eval_batch_size, self.eval_T).to(
                        self.gfn_model.device
                    )
                    _, _log_pfs, _log_pbs, _ = self.gfn_model.get_trajectory_bwd(
                        gt_xs_batch, ts, gt_log_rewards_batch
                    )
                    gt_log_pfs.append(_log_pfs)
                    gt_log_pbs.append(_log_pbs)
                gt_log_pfs = torch.cat(gt_log_pfs, dim=0)[:data_size]
                gt_log_pbs = torch.cat(gt_log_pbs, dim=0)[:data_size]
            except NotImplementedError:
                gt_xs = gt_log_rewards = gt_log_pfs = gt_log_pbs = None

        try:
            gt_log_Z = self.energy.gt_logz()
        except NotImplementedError:
            gt_log_Z = None

        metrics.update(
            density_metrics(
                log_pfs,
                log_pbs,
                log_fs,
                log_rewards,
                gt_log_pfs=gt_log_pfs,
                gt_log_pbs=gt_log_pbs,
                gt_log_rewards=gt_log_rewards,
                gt_log_Z=gt_log_Z,
            )
        )

        if gt_xs is not None and full_eval:
            # "1-Wasserstein", "2-Wasserstein", "Linear_MMD", "Poly_MMD", "RBF_MMD",
            # "Mean_MSE", "Mean_L2", "Mean_L1", "Median_MSE", "Median_L2", "Median_L1"
            metrics.update(distribution_distance_metrics(sample_xs, gt_xs))

        metrics = {f"eval/{k}": v for k, v in metrics.items()}

        ### Resample or weighted
        weights = (log_rewards + log_pbs.sum(-1) - log_pfs.sum(-1)).softmax(0)
        sample_xs_r = None
        if self.eval_resampling and full_eval:
            # We can't use `estimate_partition_function` with resampled trajectories
            # since we don't know the distribution of the resampled trajectories
            assert gt_xs is not None
            metrics_r = {}
            sampled_idx = get_sampling_func(self.resampling_strategy)(  # type: ignore
                weights, data_size, True
            )
            model_trajs_r = model_trajs[sampled_idx]
            sample_xs_r = model_trajs_r[:, -1]
            metrics_r.update(distribution_distance_metrics(sample_xs_r, gt_xs))
            metrics_r = {f"eval_resampled/{k}": v for k, v in metrics_r.items()}
            metrics.update(metrics_r)

        if self.eval_weighting and full_eval:
            assert gt_xs is not None
            metrics_w = {}
            metrics_w.update(distribution_distance_metrics(sample_xs, gt_xs, weights=weights))
            metrics_w = {f"eval_weighted/{k}": v for k, v in metrics_w.items()}
            metrics.update(metrics_w)

        buffer_xs = None
        if self.buffer is not None and len(self.buffer) > 0 and full_eval:
            assert gt_xs is not None
            buffer_xs, _ = self.buffer.sample_terminal(data_size)
            metrics_b = {}
            metrics_b.update(distribution_distance_metrics(buffer_xs, gt_xs))
            metrics_b = {f"eval_buffer/{k}": v for k, v in metrics_b.items()}
            metrics.update(metrics_b)

        if final_eval:
            metrics = {k.replace("eval", "final_eval"): v for k, v in metrics.items()}

        return metrics, model_trajs, weights, sample_xs_r, buffer_xs

    @torch.no_grad()
    def plot_step(
        self,
        model_trajs: torch.Tensor,
        weights: torch.Tensor | None = None,
        sample_xs_r: torch.Tensor | None = None,
        buffer_xs: torch.Tensor | None = None,
        plot_gt: bool = False,
    ) -> dict:
        xs = model_trajs[:, -1]
        images = visualize(self.energy, xs)
        if weights is not None:
            images.update(visualize(self.energy, xs, weights=weights, suffix="_weighted"))
        if sample_xs_r is not None:
            images.update(visualize(self.energy, sample_xs_r, suffix="_resample"))
        if buffer_xs is not None:
            images.update(visualize(self.energy, buffer_xs, suffix="_buffer"))
        if plot_gt:
            try:
                gt_xs, _ = self.energy.cached_sample(model_trajs.shape[0])
                images.update(visualize(self.energy, gt_xs, suffix="_gt"))
            except NotImplementedError:
                warnings.warn(
                    f"Ground-truth samples are not available for {self.energy.__class__.__name__}."
                    "Skipping plotting of ground-truth samples."
                )

        # Plot intermediate states
        if len(self.plot_t_idx) > 0:
            assert self.gfn_model.pred_module.conditional_flow_model
            eval_ts = self.eval_discretizer(model_trajs.shape[0], self.eval_T)
            for t_idx in self.plot_t_idx:
                inter_states = model_trajs[:, t_idx]
                assert (eval_ts[:, t_idx] == eval_ts[0, t_idx]).all()  # uniform discretizer
                eval_t = round(eval_ts[0, t_idx].item(), 3)
                inter_energy = IntermediateEnergy(self.energy, self.gfn_model, eval_t)
                images.update(visualize(inter_energy, inter_states, suffix=f"-t{eval_t}"))

        if len(self.plot_buffer_t_idx) > 0:
            assert isinstance(self.buffer, IntermediateStateBuffer)
            for t_idx in self.plot_buffer_t_idx:
                inter_states, buf_ts, _ = self.buffer.sample_timestep(model_trajs.shape[0], t_idx)
                assert (buf_ts == buf_ts[0]).all()  # uniform discretizer
                buf_t = round(buf_ts[0].item(), 3)
                inter_energy = IntermediateEnergy(self.energy, self.gfn_model, buf_t)
                images.update(visualize(inter_energy, inter_states, suffix=f"_buffer-t{buf_t}"))

        return images
