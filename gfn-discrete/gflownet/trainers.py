import math
import random
import pickle
import numpy as np
from scipy import stats
import torch
import wandb
from tqdm import tqdm
from collections import OrderedDict

from .GFNs.models import TeacherGFN
from .data import Experience


class FixSizeOrderedDict(OrderedDict):
    def __init__(self, *args, max=0, **kwargs):
        self._max = max
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        if self._max > 0:
            if len(self) > self._max:
                self.popitem(False)


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
        prioritized_buffer = FixSizeOrderedDict(max=self.args.replay_buffer_size)

        num_online = self.args.num_online_batches_per_round
        num_offline = self.args.num_offline_batches_per_round
        online_bsize = self.args.num_samples_per_online_batch
        offline_bsize = self.args.num_samples_per_offline_batch

        eval_num_samples = self.args.eval_num_samples
        print(f"Starting Training. Each round: num_online={num_online}, num_offline={num_offline}")
        total_samples = []
        accepted_samples = []
        for round_num in tqdm(
            range(self.args.num_active_learning_rounds),
            desc="Training",
            dynamic_ncols=True,
        ):
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
                        teacher_reward = torch.where(tb_delta > 0, 20 * tb_loss, tb_loss)

                        if self.args.model == "teacher":
                            assert isinstance(self.teacher, TeacherGFN)
                            self.teacher.train(
                                explore_data, torch.log(teacher_reward).detach(), round_num
                            )
                else:
                    raise ValueError(f"Unknown model: {self.args.model}")

                # Save to full dataset & prioritized buffer
                for i, exp in enumerate(explore_data):
                    if exp.x not in allXtoR:
                        allXtoR[exp.x] = exp.r

                    # We use prioritized buffer for only tb and teacher models
                    if self.args.model in ["tb", "teacher"]:
                        match self.args.offline_select:
                            case "random":
                                prioritized_buffer[exp.x] = None
                            case "reward":
                                prioritized_buffer[exp.x] = exp.r
                            case "loss":
                                prioritized_buffer[exp.x] = tb_loss[i].item()  # type: ignore
                            case "teacher_reward":
                                prioritized_buffer[exp.x] = teacher_reward[i].item()  # type: ignore
                            case "iw":
                                prioritized_buffer[exp.x] = log_iw[i].item()  # type: ignore
                            case "normalized_iw":
                                prioritized_buffer[exp.x] = normalized_iw[i].item()  # type: ignore
                            case _:
                                raise ValueError(
                                    f"Unknown offline select: {self.args.offline_select}"
                                )

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
                else:
                    offline_xs = self.select_offline_xs(prioritized_buffer, offline_bsize)
                    offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR)
                    for _ in range(self.args.offline_num_steps_per_batch):
                        log_iw = self.model.train(offline_dataset)

                        # Train teacher model with teacher reward
                        if self.args.model == "teacher":
                            tb_delta = log_iw - self.model.logZ
                            tb_loss = tb_delta**2
                            teacher_reward = torch.where(tb_delta > 0, 20 * tb_loss, tb_loss)
                            assert isinstance(self.teacher, TeacherGFN)
                            self.teacher.train(
                                offline_dataset, torch.log(teacher_reward).detach(), round_num
                            )

                if self.args.offline_select in ["loss", "teacher_reward"]:
                    for i, exp in enumerate(offline_dataset):
                        if self.args.offline_select == "loss":
                            prioritized_buffer[exp.x] = None  # FIXME with new buffer
                        elif self.args.offline_select == "teacher_reward":
                            prioritized_buffer[exp.x] = None  # FIXME with new buffer

            if round_num and (
                round_num % self.args.eval_every_x_active_rounds == 0
                or round_num == self.args.num_active_learning_rounds - 1
            ):
                print(f"Evaluating round {round_num+1} ...")
                self.model.policy_fwd.eval()
                self.model.policy_back.eval()
                with torch.no_grad():
                    onpolicy_samples = self.model.batch_fwd_sample(eval_num_samples, epsilon=0)
                    results = self.evaluate(round_num, onpolicy_samples, allXtoR)
                self.model.policy_fwd.train()
                self.model.policy_back.train()
                wandb.log(results)

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

    def select_offline_xs(self, buffer: FixSizeOrderedDict, batch_size: int):
        if self.args.offline_select == "random":
            return random.choices(list(buffer.keys()), k=batch_size)
        else:
            return self.__biased_sample_xs(buffer, batch_size)

    def __biased_sample_xs(self, buffer: FixSizeOrderedDict, batch_size: int):
        """Select xs for offline training. Returns List of [State].
        Draws 50% from top 10% of priority, and 50% from bottom 90%.
        """
        if len(buffer) < 10:
            return []
        priority = np.array(list(buffer.values()))
        threshold = np.percentile(priority, 90)
        top_xs = [x for x, p in buffer.items() if p >= threshold]
        bottom_xs = [x for x, p in buffer.items() if p <= threshold]
        sampled_xs = random.choices(top_xs, k=batch_size // 2) + random.choices(
            bottom_xs, k=batch_size // 2
        )
        return sampled_xs

    def offline_PB_traj_sample(self, offline_xs, allXtoR):
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

        for k, v in results.items():
            if isinstance(v, float):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        return results
