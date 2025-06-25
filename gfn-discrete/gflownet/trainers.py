import math
import random
import pickle
from typing import cast
import numpy as np
from scipy import stats
import torch
import wandb
from tqdm import trange

from gflownet.GFNs.models import TeacherGFN
from gflownet.data import Experience
from gflownet.buffer import TerminalStateBuffer
from gflownet.utils.sampling_utils import get_sampling_func
from gflownet.MDPs.basemdp import BaseState


class Trainer:
    def __init__(self, args, model, mdp, teacher=None):
        self.args = args
        self.model = model
        self.mdp = mdp
        self.teacher = teacher
        return

    """
    Training
    """

    def learn(self) -> None:
        """Main learning training loop.
        Each learning round:
          Each online batch:
            sample a new dataset using exploration policy.
          Each offline batch:
            resample batch from full historical dataset

        To learn on fixed dataset only: Set 0 online batches per round,
        and provide initial dataset.

        dataset = List of [Experience]
        """
        allXtoR = dict()
        prioritized_buffer = TerminalStateBuffer(
            buffer_size=self.args.replay_buffer_size,
            device=self.args.device,
            prioritization=self.args.prioritization,
            sampling_func=get_sampling_func(self.args.buffer_sampling, self.args.rank_k),
            target_ess=self.args.target_ess,
            smoothing_strategy=self.args.smoothing_strategy,
        )

        num_online = self.args.num_online_batches_per_round
        num_offline = self.args.num_offline_batches_per_round
        online_bsize = self.args.num_samples_per_online_batch
        offline_bsize = self.args.num_samples_per_offline_batch

        eval_num_samples = self.args.eval_num_samples
        print(f"Starting Training. Each round: num_online={num_online}, num_offline={num_offline}")
        total_samples = []
        accepted_samples = []
        pbar = trange(self.args.num_active_learning_rounds, dynamic_ncols=True)
        for round_num in pbar:
            pbar.set_description("Training")
            # Online training - skip first if initial dataset was provided
            for _ in range(num_online):
                if self.args.ls:
                    with torch.no_grad():
                        explore_data = self.model.batch_fwd_sample_ls(
                            online_bsize,
                            epsilon=self.args.explore_epsilon,
                            k=self.args.k,
                            i=self.args.i,
                            deterministic=self.args.deterministic,
                        )
                elif self.args.model == "mars":
                    with torch.no_grad():
                        if len(total_samples) == 0:
                            explore_data = self.model.batch_fwd_sample(
                                online_bsize, epsilon=self.args.explore_epsilon
                            )
                        else:
                            explore_data, accepted_data = self.model.batch_fwd_sample(
                                online_bsize,
                                epsilon=self.args.explore_epsilon,
                                explore_data=explore_data,
                            )
                            accepted_samples.extend(accepted_data)
                elif self.args.model == "teacher":
                    assert self.teacher is not None
                    with torch.no_grad():
                        if round_num % 3 == 0 or round_num % 3 == 1:
                            model_or_teacher = self.model
                        else:
                            model_or_teacher = self.teacher
                        explore_data = model_or_teacher.batch_fwd_sample(
                            online_bsize, epsilon=self.args.explore_epsilon
                        )
                else:
                    with torch.no_grad():
                        explore_data = self.model.batch_fwd_sample(
                            online_bsize, epsilon=self.args.explore_epsilon
                        )

                # Train on online dataset
                if self.args.model in ["a2c", "sql", "mars"]:
                    pass
                elif self.args.model == "ppo":
                    # As ppo is on-policy algorithm, we double online training steps
                    for _ in range(self.args.online_num_steps_per_batch * 2):
                        self.model.train(explore_data)
                elif self.args.model in ["tb", "teacher"]:
                    for _ in range(self.args.online_num_steps_per_batch):
                        log_iw = self.model.train(explore_data)
                        # Train teacher model with teacher reward
                        normalized_iw = log_iw.softmax(dim=0)
                        tb_delta = log_iw - self.model.logZ
                        tb_loss = tb_delta**2

                        if self.args.model == "teacher":
                            teacher_reward = torch.log(
                                torch.where(tb_delta > 0, 20 * tb_loss, tb_loss) + 1.001
                            )  # Note: this definition of teacher reward is different from the one in the paper
                            assert isinstance(self.teacher, TeacherGFN)
                            self.teacher.train(
                                explore_data, torch.log(teacher_reward).detach(), round_num
                            )

                    # Save experiences to prioritized buffer
                    states = np.array([exp.x for exp in explore_data])
                    log_rs = torch.stack([exp.logr for exp in explore_data])
                    data_dict = {"states": states, "log_fs": log_rs}
                    match self.args.prioritization:
                        case "loss":
                            data_dict["losses"] = tb_loss
                        case "iw":
                            data_dict["log_iws"] = log_iw
                        case "normalized_iw":
                            data_dict["normalized_iws"] = normalized_iw
                        case _:
                            raise ValueError(f"Unknown offline select: {self.args.prioritization}")
                    prioritized_buffer.add(**data_dict)
                else:
                    raise ValueError(f"Unknown model: {self.args.model}")

                for i, exp in enumerate(explore_data):
                    if exp.x not in allXtoR:
                        allXtoR[exp.x] = exp.r

                total_samples.extend(explore_data)

            # Offline training
            for _ in range(num_offline):
                if self.args.model == "a2c" or self.args.model == "sql":
                    # we do not use PRT for RL-based methods
                    # As A2C and SQL are off-policy algorithm, we double offline training steps
                    offline_dataset = random.choices(total_samples, k=offline_bsize)
                    for _ in range(self.args.offline_num_steps_per_batch * 2):
                        self.model.train(offline_dataset)
                elif self.args.model == "ppo":
                    pass
                elif self.args.model == "mars":
                    if len(accepted_samples) >= offline_bsize:
                        offline_dataset = random.choices(accepted_samples, k=offline_bsize)
                        for _ in range(self.args.offline_num_steps_per_batch):
                            self.model.train(offline_dataset)
                elif self.args.model in ["tb", "teacher"]:
                    offline_xs, offline_log_rs, offline_indices = prioritized_buffer.sample(
                        offline_bsize
                    )
                    offline_dataset = self.offline_PB_traj_sample(offline_xs.tolist(), allXtoR)
                    for _ in range(self.args.offline_num_steps_per_batch):
                        log_iw = self.model.train(offline_dataset)

                        # Train teacher model with teacher reward
                        if self.args.model == "teacher":
                            tb_delta = log_iw - self.model.logZ
                            tb_loss = tb_delta**2
                            teacher_reward = torch.log(
                                torch.where(tb_delta > 0, 20 * tb_loss, tb_loss) + 1.001
                            )  # Note: this definition of teacher reward is different from the one in the paper
                            self.teacher.train(  # type: ignore
                                offline_dataset, torch.log(teacher_reward).detach(), round_num
                            )

                    if self.args.prioritization == "loss":
                        prioritized_buffer.update(
                            offline_indices, losses=cast(torch.Tensor, tb_loss)
                        )
                else:
                    raise ValueError(f"Unknown model: {self.args.model}")

            if round_num and (
                round_num % self.args.eval_every_x_active_rounds == 0
                or round_num == self.args.num_active_learning_rounds - 1
            ):
                pbar.set_description("Evaluating")
                self.model.policy_fwd.eval()
                self.model.policy_back.eval()
                with torch.no_grad():
                    onpolicy_samples = self.model.batch_fwd_sample(eval_num_samples, epsilon=0)
                    results = self.evaluate(round_num, onpolicy_samples, allXtoR)
                self.model.policy_fwd.train()
                self.model.policy_back.train()
                wandb.log(results)
                pbar_dict = {
                    k.replace("_", "").upper(): v
                    for k, v in results.items()
                    if k in ["elbo", "eubo", "all_num_modes"]
                }
                pbar_dict["logZ"] = self.model.logZ.item()
                pbar.set_postfix(pbar_dict)

            if round_num and (
                round_num % self.args.save_every_x_active_rounds == 0
                or round_num == self.args.num_active_learning_rounds - 1
            ):
                self.model.save_params(
                    self.args.saved_models_dir
                    + self.args.run_name
                    + "/"
                    + f"round_{round_num+1}.pth"
                )
                with open(
                    self.args.saved_models_dir
                    + self.args.run_name
                    + "/"
                    + f"round_{round_num+1}_sample.pkl",
                    "wb",
                ) as f:
                    pickle.dump(total_samples, f)

        print("Finished training.")
        self.model.save_params(self.args.saved_models_dir + self.args.run_name + "/" + "final.pth")
        with open(
            self.args.saved_models_dir + self.args.run_name + "/" + f"final_sample.pkl", "wb"
        ) as f:
            pickle.dump(total_samples, f)

    """
    Offline training
    """

    def offline_PB_traj_sample(
        self, offline_xs: list[BaseState], allXtoR: dict[BaseState, float] | None
    ):
        """Sample trajectories for x using P_B, for offline training with TB.
        Returns List of [Experience].
        """
        if allXtoR is None:
            offline_rs = [self.mdp.reward(x) for x in offline_xs]
        else:
            offline_rs = [allXtoR[x] for x in offline_xs]

        # Not subgfn: sample trajectories from backward policy
        with torch.no_grad():
            offline_trajs = self.model.batch_back_sample(offline_xs)

        offline_dataset = [
            Experience(
                traj=traj,
                x=x,
                r=r,
                logr=torch.log(torch.tensor(r, dtype=torch.float32, device=self.args.device)),
            )
            for traj, x, r in zip(offline_trajs, offline_xs, offline_rs)
        ]
        return offline_dataset

    def evaluate(self, round_num, samples, allXtoR):
        log_r = torch.stack([exp.logr for exp in samples])

        # Metric 1. Gap Between True Expected Reward and Estimated Expected Reward
        estimated_expected_reward = log_r.exp().mean().item()
        estimated_expected_reward_gap = np.abs(estimated_expected_reward - self.mdp.expected_reward)

        # Metric 2. Gap Between True LogZ and Estimated LogZ / elbo and eubo
        fwd_logp = self.model.batch_traj_fwd_logp(samples)
        if self.args.model == "gafn":
            back_logp = self.model.batch_traj_bwd_logp(samples, evaluate=True)
        else:
            back_logp = self.model.batch_traj_bwd_logp(samples)
        iw_elbo = torch.logsumexp(log_r + back_logp - fwd_logp, 0).item() - math.log(len(samples))
        elbo = (log_r + back_logp - fwd_logp).mean().item()

        fwd_logp_true_all = []
        eubo = []
        bsz = self.args.num_samples_per_online_batch * 10  # Can use bigger batch size for testing
        for i in range(0, len(self.mdp.true_samples), bsz):
            true_samples = self.offline_PB_traj_sample(self.mdp.true_samples[i : i + bsz], None)
            true_log_r = torch.stack([exp.logr for exp in true_samples])
            fwd_logp_true = self.model.batch_traj_fwd_logp(true_samples)
            if self.args.model == "gafn":
                back_logp_true = self.model.batch_traj_bwd_logp(true_samples, evaluate=True)
            else:
                back_logp_true = self.model.batch_traj_bwd_logp(true_samples)
            fwd_logp_true_all.append(fwd_logp_true)
            eubo.append(true_log_r + back_logp_true - fwd_logp_true)
        eubo = torch.cat(eubo).mean().item()

        # Metric 3. Pearson Correlation Coefficient
        pearson_corr = stats.pearsonr(
            torch.cat(fwd_logp_true_all).cpu().detach().numpy(),
            np.log(np.array([self.mdp.reward(x) for x in self.mdp.true_samples])),
        )[0]

        # Metric 4. Number of modes
        onpolicy_xs = set([exp.x for exp in samples])
        onpolicy_num_modes = 0
        for x in onpolicy_xs:
            if x.content in self.mdp.modes:
                onpolicy_num_modes += 1

        all_num_modes = 0
        for x, _ in allXtoR.items():
            if x.content in self.mdp.modes:
                all_num_modes += 1

        results = {
            "round_num": round_num,
            "estimated_expected_reward": estimated_expected_reward,
            "estimated_expected_reward_gap": estimated_expected_reward_gap,
            "iw_elbo": iw_elbo,
            "elbo": elbo,
            "eubo": eubo,
            "pearson_corr": pearson_corr,
            "onpolicy_num_modes": onpolicy_num_modes,
            "all_num_modes": all_num_modes,
        }

        return results
