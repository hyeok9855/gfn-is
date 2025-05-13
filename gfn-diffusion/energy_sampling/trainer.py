from typing import Callable, Literal

import torch

from buffers import BaseBuffer, IntermediateStateBuffer, TerminalStateBuffer
from energies import BaseEnergy, IntermediateEnergy
from losses import cal_subtb_coef_matrix, get_loss
from models import GFN
from utils.eval_utils import density_metrics, distribution_distance_metrics
from utils.misc_utils import linear_annealing
from utils.plot_utils import visualize
from utils.sampling_utils import get_sampling_func
from utils.train_utils import get_normalized_weights


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
        training_mode: Literal["fwd", "bwd", "both"],
        bwd_from: Literal["energy", "buffer"],
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
        aux_target: Literal["target", "loss"],
        target_ess: float,
        smoothing_strategy: Literal["temper", "clip_above", "clip_below"],
        epsilon: float,
        anneal_epsilon: bool,
        invtemp: float,
        invtemp_anneal: bool,
        eval_discretizer: Callable[[int, int], torch.Tensor],
        eval_T: int,
        eval_weighting: bool,
        eval_resampling: bool,
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
        self.training_mode = training_mode
        self.bwd_from = bwd_from
        if bwd_to_fwd_ratio < 1:
            self.bwd_to_fwd_ratio = -1.0
            self.fwd_to_bwd_ratio = round(1 / bwd_to_fwd_ratio)
        else:
            self.bwd_to_fwd_ratio = bwd_to_fwd_ratio
            self.fwd_to_bwd_ratio = -1.0
        self.buffer = buffer
        self.buffer_save_interval = buffer_save_interval
        self.prefill_epochs = prefill_epochs

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
        self.aux_target = aux_target
        self.target_ess = target_ess
        self.smoothing_strategy = smoothing_strategy

        # Misc
        self.epsilon = epsilon
        self.anneal_epsilon = anneal_epsilon
        self.invtemp = invtemp
        self.invtemp_anneal = invtemp_anneal

        # Eval and Plot
        self.eval_discretizer = eval_discretizer
        self.eval_T = eval_T
        self.eval_weighting = eval_weighting
        self.eval_resampling = eval_resampling
        self.plot_t_idx = plot_t_idx
        self.plot_buffer_t_idx = plot_buffer_t_idx

    @property
    def weighting(self) -> bool:
        if self._weighting:
            if self.alternating:
                self._alternating_flag = not self._alternating_flag
                return self._alternating_flag
            else:
                return True
        else:
            return False

    @property
    def resampling(self) -> bool:
        if self._resampling:
            if self.alternating:
                self._alternating_flag = not self._alternating_flag
                return self._alternating_flag
            else:
                return True
        else:
            return False

    def train_step(self, it: int) -> float:
        self.gfn_model.train()
        self.energy.invtemp = (
            self.invtemp
            if not self.invtemp_anneal
            else linear_annealing(
                it // 100, int(0.8 * self.n_epochs) // 100, self.invtemp, 1.0, descending=False
            )
        )

        if self.training_mode == "bwd":
            loss = self.bwd_train_step()
        elif self.training_mode == "fwd":
            loss = self.fwd_train_step(it)
        else:  # both

            if (
                (self.bwd_to_fwd_ratio > 0 and it % (self.bwd_to_fwd_ratio + 1) == 0)
                or (self.bwd_to_fwd_ratio < 0 and it % (self.fwd_to_bwd_ratio + 1) != 0)
            ) or it < self.prefill_epochs:
                loss = self.fwd_train_step(it)
            else:
                loss = self.bwd_train_step()

        if it < self.prefill_epochs:
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

        normalized_iws_0t = None
        if (
            (self.buffer is not None and self.buffer.prioritization == "normalized_iw")
            or self.weighting
            or self.resampling
        ):
            normalized_iws_0t = get_normalized_weights(
                self.batch_size,
                ts,
                log_fs,
                log_pbs,
                log_pfs_exp,
                self.device,
                self.aux_target,
                self.loss_type,
                self.target_ess,
                self.smoothing_strategy,
                losses,
            )

        # Add data to buffer
        if self.buffer is not None:
            data_dict = {"states": states[:, 1:], "log_fs": log_fs[:, 1:]}  # (bs, T) both
            if self.buffer.prioritization == "loss":
                assert isinstance(self.buffer, TerminalStateBuffer)
                data_dict["losses"] = losses.unsqueeze(1)  # (bs, 1)
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

        return loss

    def bwd_train_step(self) -> torch.Tensor:
        sub_logvar_params = {}

        if self.bwd_from == "energy":
            raise NotImplementedError("Training from energy is not used for this project.")

        elif self.bwd_from == "buffer":
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
                sub_logvar_params = {"sublogvar_K": self.sublogvar_K, "ts": ts, "curr_t": buf_ts}

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
            images = self.plot_step(model_trajs, weights, sample_xs_r, buffer_xs)
            metrics.update(images)
        return metrics

    @torch.no_grad()
    def eval_step(
        self,
        data_size: int,
        full_eval: bool = False,
        final_eval: bool = False,
    ) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        self.gfn_model.eval()

        if final_eval:
            full_eval = True

        metrics = {}

        with torch.no_grad():
            divisible = data_size % self.batch_size == 0
            n_epochs = data_size // self.batch_size + (1 if not divisible else 0)

            model_trajs, log_pfs, log_pbs, log_fs, log_rewards = [], [], [], [], []

            for i in range(n_epochs):
                if i == n_epochs - 1 and not divisible:
                    bsz = data_size - self.batch_size * (n_epochs - 1)
                else:
                    bsz = self.batch_size

                init_state = torch.zeros(bsz, self.energy.ndim).to(self.gfn_model.device)
                ts = self.eval_discretizer(bsz, self.eval_T).to(self.gfn_model.device)
                _model_trajs, _log_pfs, _log_pbs, _log_fs, _ = self.gfn_model.get_trajectory_fwd(
                    init_state, ts, epsilon=0.0, pis=self.loss_type == "pis"
                )
                _log_rewards = self.energy.log_reward(_model_trajs[:, -1], temper=False)
                model_trajs.append(_model_trajs)
                log_pfs.append(_log_pfs)
                log_pbs.append(_log_pbs)
                log_fs.append(_log_fs)
                log_rewards.append(_log_rewards)
            model_trajs = torch.cat(model_trajs, dim=0)
            sample_xs = model_trajs[:, -1]
            log_pfs = torch.cat(log_pfs, dim=0)
            log_pbs = torch.cat(log_pbs, dim=0)
            log_fs = torch.cat(log_fs, dim=0)
            log_rewards = torch.cat(log_rewards, dim=0)

            try:
                gt_xs, gt_log_rewards = self.energy.cached_sample(data_size)
                gt_log_pfs, gt_log_pbs = [], []
                for i in range(n_epochs):
                    gt_xs_batch = gt_xs[i * self.batch_size : (i + 1) * self.batch_size]
                    ts = self.eval_discretizer(gt_xs_batch.shape[0], self.eval_T).to(
                        self.gfn_model.device
                    )
                    gt_log_rewards_batch = gt_log_rewards[
                        i * self.batch_size : (i + 1) * self.batch_size
                    ]
                    _, _log_pfs, _log_pbs, _ = self.gfn_model.get_trajectory_bwd(
                        gt_xs_batch, ts, gt_log_rewards_batch
                    )
                    gt_log_pfs.append(_log_pfs)
                    gt_log_pbs.append(_log_pbs)
                gt_log_pfs = torch.cat(gt_log_pfs, dim=0)
                gt_log_pbs = torch.cat(gt_log_pbs, dim=0)
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
                weights, self.batch_size, True
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
    ) -> dict:
        images = visualize(self.energy, model_trajs[:, -1])
        if weights is not None:
            images.update(
                visualize(self.energy, model_trajs[:, -1], weights=weights, suffix="_weighted")
            )
        if sample_xs_r is not None:
            images.update(visualize(self.energy, sample_xs_r, suffix="_resample"))
        if buffer_xs is not None:
            images.update(visualize(self.energy, buffer_xs, suffix="_buffer"))

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
