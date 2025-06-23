"""
TFBind8
Oracle
Start from scratch
No proxy
"""

import os
import copy, pickle
import numpy as np
from tqdm import tqdm
import torch
from polyleven import levenshtein
from argparse import Namespace

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqarmdp


def dynamic_inherit_mdp(base, args):

    class TFBind8MDP(base):
        def __init__(self, args):
            super().__init__(args, alphabet=list("0123"), forced_stop_len=args.forced_stop_len)
            self.args = args

            # Read from file
            print(f"Loading data ...")
            with open("datasets/tfbind8/tfbind8-exact-v0-all.pkl", "rb") as f:
                oracle_d = pickle.load(f)

            munge = lambda x: "".join([str(c) for c in list(x)])
            self.oracle = {
                self.state(munge(x), is_leaf=True): float(y)
                for x, y in zip(oracle_d["x"], oracle_d["y"])
            }

            # Scale rewards
            self.scaled_oracle = copy.copy(self.oracle)
            py = np.array(list(self.scaled_oracle.values()))

            self.SCALE_REWARD_MIN = args.scale_reward_min
            self.SCALE_REWARD_MAX = args.scale_reward_max
            self.REWARD_EXP = args.beta

            py = np.maximum(py, self.SCALE_REWARD_MIN)
            py = py**self.REWARD_EXP
            self.scale = self.SCALE_REWARD_MAX / max(py)
            py = np.maximum(py * self.scale, 1e-20)

            self.scaled_oracle = {x: y for x, y in zip(self.scaled_oracle.keys(), py)}

            # Rewards
            self.rs_all = [y for x, y in self.scaled_oracle.items()]

            # Modes
            if args.mode_metric == "default":
                with open("datasets/tfbind8/modes_tfbind8.pkl", "rb") as f:
                    modes = pickle.load(f)
                self.modes = set([munge(x) for x in modes])
            elif args.mode_metric == "threshold":
                if os.path.exists(f"datasets/tfbind8/modes_percentile_{args.mode_percentile}.pkl"):
                    with open(
                        f"datasets/tfbind8/modes_percentile_{args.mode_percentile}.pkl", "rb"
                    ) as f:
                        self.modes = pickle.load(f)
                else:
                    mode_percentile = args.mode_percentile
                    self.mode_r_threshold = np.percentile(py, 100 * (1 - mode_percentile))
                    num_modes = int(len(self.scaled_oracle) * mode_percentile)
                    sorted_xs = sorted(self.scaled_oracle, key=self.scaled_oracle.get)
                    self.modes = set([x.content for x in sorted_xs[-num_modes:]])
                    with open(
                        f"datasets/tfbind8/modes_percentile_{args.mode_percentile}.pkl", "wb"
                    ) as f:
                        pickle.dump(self.modes, f)
            elif args.mode_metric == "hammingball":
                if os.path.exists(
                    f"datasets/tfbind8/modes_hammingball_dist{args.mode_hammingball_dist}_percentile_{args.mode_percentile}.pkl"
                ):
                    with open(
                        f"datasets/tfbind8/modes_hammingball_dist{args.mode_hammingball_dist}_percentile_{args.mode_percentile}.pkl",
                        "rb",
                    ) as f:
                        self.modes = pickle.load(f)
                else:
                    mode_percentile = args.mode_percentile
                    self.mode_r_threshold = np.percentile(py, 100 * (1 - mode_percentile))
                    self.mode_hammingball_dist = args.mode_hammingball_dist

                    print(
                        f"Computing modes with hamming ball distance {self.mode_hammingball_dist}..."
                    )
                    self.modes = set()
                    for x, y in tqdm(self.scaled_oracle.items()):
                        y = self.scaled_oracle[x]
                        if y >= self.mode_r_threshold:
                            if len(self.modes) == 0:
                                self.modes.add(x)
                            else:
                                flag = False
                                for mode in self.modes:
                                    edit_dist = levenshtein(x.content, mode.content)
                                    if edit_dist <= self.mode_hammingball_dist:
                                        flag = True
                                        break
                                if not flag:
                                    self.modes.add(x)
                    self.modes = set([x.content for x in self.modes])
                    with open(
                        f"datasets/tfbind8/modes_hammingball_dist{args.mode_hammingball_dist}_percentile_{args.mode_percentile}.pkl",
                        "wb",
                    ) as f:
                        pickle.dump(self.modes, f)

            print(f"Mode metric: {args.mode_metric}\tFound num modes: {len(self.modes)}")

            # compute true expected reward and logz
            py = np.array(list(self.oracle.values()))
            py = np.maximum(py, self.SCALE_REWARD_MIN)
            log_py = np.log(py)

            self.logZ = torch.logsumexp(torch.from_numpy(log_py * self.REWARD_EXP), 0).item()
            logr_square = torch.logsumexp(torch.from_numpy(log_py * 2), 0).item()
            self.expected_reward = np.exp(logr_square - self.logZ)

            log_py_scaled = log_py * self.REWARD_EXP
            log_scale = np.log(self.SCALE_REWARD_MAX) - max(log_py_scaled)
            log_py_scaled = log_py_scaled + log_scale
            self.logZ_scaled = torch.logsumexp(
                torch.from_numpy(log_py_scaled).float(), dim=0
            ).item()
            logr_square_scaled = torch.logsumexp(
                torch.from_numpy(log_py_scaled + log_py).float(), dim=0
            ).item()
            self.expected_reward_scaled = np.exp(logr_square_scaled - self.logZ_scaled)

            print(
                f"Beta: {self.REWARD_EXP}\tExpected reward: {self.expected_reward:.2f}\tLogZ: {self.logZ:.2f}"
            )

            # generate samples from true distribution
            # all_samples = list(self.oracle.keys())
            # true_dist = np.exp(log_py * self.REWARD_EXP - self.logZ)
            # true_samples_idx = np.random.choice(len(self.oracle), size=args.eval_num_samples, p=true_dist)
            # self.true_samples = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx]

            # true_dist_scaled = np.exp(log_py_scaled - self.logZ_scaled)
            # true_samples_idx_scaled = np.random.choice(len(self.oracle), size=args.eval_num_samples, p=true_dist_scaled)
            # self.true_samples_scaled = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx_scaled]

            all_samples = oracle_d["x"]
            true_samples_idx = []
            logprobs = log_py * self.REWARD_EXP - self.logZ
            for _ in tqdm(range(self.args.eval_num_samples)):
                gumbels = np.random.gumbel(size=(len(self.oracle)))
                true_samples_idx.append(np.argmax(logprobs + gumbels))
            self.true_samples = [
                self.state(munge(all_samples[i]), is_leaf=True) for i in true_samples_idx
            ]

            true_samples_idx_scaled = []
            logprobs_scaled = log_py_scaled - self.logZ_scaled
            for _ in tqdm(range(self.args.eval_num_samples)):
                gumbels = np.random.gumbel(size=(len(self.oracle)))
                true_samples_idx_scaled.append(np.argmax(logprobs_scaled + gumbels))
            self.true_samples_scaled = [
                self.state(munge(all_samples[i]), is_leaf=True) for i in true_samples_idx_scaled
            ]

        # Core
        def reward(self, x):
            assert x.is_leaf, "Error: Tried to compute reward on non-leaf node."
            return self.scaled_oracle[x]

        def real_reward(self, x):
            assert x.is_leaf, "Error: Tried to compute reward on non-leaf node."
            return np.maximum(self.oracle[x], self.SCALE_REWARD_MIN)

        def is_mode(self, x, r):
            return x in self.modes

        def unnormalize(self, r):
            r = r / self.scale
            r = r ** (1 / self.REWARD_EXP)
            return r

        """
        Interpretation & visualization
        """

        def dist_func(self, state1, state2):
            """States are SeqPAState or SeqInsertState objects."""
            return levenshtein(state1.content, state2.content)

    return TFBind8MDP(args)


def main(args: Namespace):
    print("Running experiment TFBind8 ...")

    if args.mdp_style == "pa":
        base = seqpamdp.SeqPrependAppendMDP
        actorclass = seqpamdp.SeqPAActor
    elif args.mdp_style == "autoregressive":
        base = seqarmdp.SeqAutoregressiveMDP
        actorclass = seqarmdp.SeqARActor

    if args.model == "gafn":
        mdp = dynamic_inherit_mdp(base, args)
        actor = actorclass(args, mdp)
        rnd_target = actorclass(args, mdp)
        rnd_predict = actorclass(args, mdp)
        model = models.make_model(args, mdp, actor, rnd_target=rnd_target, rnd_predict=rnd_predict)
        trainer = trainers.Trainer(args, model, mdp, actor)
    elif args.model == "teacher":
        mdp_student = dynamic_inherit_mdp(base, args)
        mdp_teacher = dynamic_inherit_mdp(base, args)
        actor_student = actorclass(args, mdp_student)
        actor_teacher = actorclass(args, mdp_teacher)
        model_teacher, model_student = models.make_teacher_student_model(
            args, mdp_student, mdp_teacher, actor_student, actor_teacher
        )
        trainer = trainers.Trainer(args, model_student, mdp_student, teacher=model_teacher)
    else:
        mdp = dynamic_inherit_mdp(base, args)

        actor = actorclass(args, mdp)
        model = models.make_model(args, mdp, actor)
        trainer = trainers.Trainer(args, model, mdp)

    trainer.learn()
    return


def number_of_modes(args: Namespace):
    print("Running evaluation TFBind8 ...")

    if args.mdp_style == "pa":
        base = seqpamdp.SeqPrependAppendMDP
    elif args.mdp_style == "autoregressive":
        base = seqarmdp.SeqAutoregressiveMDP
    mdp = dynamic_inherit_mdp(base, args)

    # load model checkpoint
    ckpt_path = args.saved_models_dir + args.run_name
    with open(ckpt_path + "/" + f"final_sample.pkl", "rb") as f:
        generated_samples = pickle.load(f)

    unique_modes = set()
    batch_size = args.num_samples_per_online_batch
    number_of_modes = np.zeros((len(generated_samples) // batch_size,))
    with tqdm(total=len(generated_samples)) as pbar:
        for i in range(0, len(generated_samples), batch_size):
            for exp in generated_samples[i : i + batch_size]:
                if mdp.is_mode(exp.x, exp.r) and exp.x.content not in unique_modes:
                    unique_modes.add(exp.x.content)
                    number_of_modes[i // batch_size] += 1
            pbar.update(batch_size)
            pbar.set_postfix(number_of_modes=np.sum(number_of_modes))
    print(np.sum(number_of_modes))
    np.savez_compressed(ckpt_path + "/" + f"number_of_modes.npz", modes=number_of_modes)
