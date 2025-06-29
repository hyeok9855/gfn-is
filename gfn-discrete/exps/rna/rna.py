"""
RNA
from flexs
Start from scratch
"""

from itertools import product
import os
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from polyleven import levenshtein

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqarmdp
from gflownet.utils.misc_utils import scale_rewards

import flexs


def dynamic_inherit_mdp(base, args):

    class RNAMDP(base):
        def __init__(self, args):
            super().__init__(
                args, alphabet=["U", "C", "G", "A"], forced_stop_len=args.forced_stop_len
            )
            self.args = args
            self.rna_task = args.rna_task
            self.rna_length = args.rna_length
            prefix = f"datasets/rna/L{self.rna_length}_RNA{self.rna_task}"
            allpreds_file = f"{prefix}_allpreds.pkl"

            print(f"Loading oracle ...")
            problem = flexs.landscapes.rna.registry()[f"L{self.rna_length}_RNA{self.rna_task}"]
            self.oracle = flexs.landscapes.RNABinding(**problem["params"])

            with open(allpreds_file, "rb") as f:
                all_rewards = pickle.load(f)
            scaled_rewards = scale_rewards(
                all_rewards, args.beta, args.scale_reward_min, args.scale_reward_max
            )
            assert min(scaled_rewards) > 0

            self._min, self._max = min(all_rewards), max(all_rewards)
            self.scaled_oracle = lambda x: (
                (np.log(1 + np.exp(self.oracle.get_fitness([x]).item())) ** args.beta - self._min)
                / (self._max - self._min)
                * (args.scale_reward_max - args.scale_reward_min)
                + args.scale_reward_min
            )

            assert args.mode_metric == "hammingball"
            filename = f"{prefix}_modes_hammingball_dist{args.mode_hammingball_dist}_percentile_{args.mode_percentile}.pkl"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    self.modes = pickle.load(f)
            else:
                print("Loading data...")
                self.mode_r_threshold = np.percentile(all_rewards, 100 * (1 - args.mode_percentile))

                print(f"Computing modes with hamming ball distance {args.mode_hammingball_dist}...")
                self.modes = set()
                all_samples = [
                    "".join(x) for x in product(["A", "U", "C", "G"], repeat=self.rna_length)
                ]
                for x, r in tqdm(zip(all_samples, all_rewards)):
                    if r >= self.mode_r_threshold:
                        if len(self.modes) == 0:
                            self.modes.add(x)
                        else:
                            flag = False
                            for mode in self.modes:
                                edit_dist = levenshtein(x, mode)
                                if edit_dist <= self.mode_hammingball_dist:
                                    flag = True
                                    break
                            if not flag:
                                self.modes.add(x)
                with open(filename, "wb") as f:
                    pickle.dump(self.modes, f)
                del all_samples

            print(f"Mode metric: {args.mode_metric}\tFound num modes: {len(self.modes)}")

            log_rewards = np.log(scaled_rewards)
            self.logZ = torch.logsumexp(torch.from_numpy(log_rewards), 0).item()
            log_rewards_square = torch.logsumexp(torch.from_numpy(log_rewards * 2), 0).item()
            self.expected_reward = np.exp(log_rewards_square - self.logZ)

            print(
                f"Beta: {args.beta}\tExpected reward: {self.expected_reward:.2f}\tLogZ: {self.logZ:.2f}"
            )

            true_samples_filename = f"{prefix}_samples_beta{args.beta}_n{args.eval_num_samples}.pkl"
            if os.path.exists(true_samples_filename):
                with open(true_samples_filename, "rb") as f:
                    self.true_samples = pickle.load(f)
            else:
                # generate samples from true distribution
                all_samples = [
                    "".join(x) for x in product(["A", "U", "C", "G"], repeat=self.rna_length)
                ]
                true_dist = np.exp(log_rewards - self.logZ)
                true_samples_idx = np.random.choice(
                    len(all_samples), size=args.eval_num_samples, p=true_dist, replace=True
                )
                self.true_samples = [
                    self.state(all_samples[i], is_leaf=True) for i in true_samples_idx
                ]

                with open(true_samples_filename, "wb") as f:
                    pickle.dump(self.true_samples, f)
                del all_samples, true_dist, true_samples_idx
            del all_rewards, scaled_rewards, log_rewards, log_rewards_square

        # Core
        def reward(self, x):
            assert x.is_leaf, "Error: Tried to compute reward on non-leaf node."
            return self.scaled_oracle(x.content)

        def is_mode(self, x, r):
            # if self.mode_metric == "threshold":
            #   return r >= self.mode_r_threshold
            # else:
            return x.content in self.modes

        """
        Interpretation & visualization
        """

        def dist_func(self, state1, state2):
            """States are SeqPAState or SeqInsertState objects."""
            return levenshtein(state1.content, state2.content)

    return RNAMDP(args)


def main(args):
    print("Running experiment RNA ...")

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
        trainer = trainers.Trainer(args, model, mdp)
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
