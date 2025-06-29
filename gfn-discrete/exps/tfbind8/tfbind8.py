"""
TFBind8
Oracle
Start from scratch
No proxy
"""

import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
from polyleven import levenshtein
from argparse import Namespace

from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqarmdp
from gflownet.trainers import Trainer
from gflownet.utils.misc_utils import scale_rewards


def dynamic_inherit_mdp(base, args):

    class TFBind8MDP(base):
        def __init__(self, args):
            super().__init__(args, alphabet=list("0123"), forced_stop_len=args.forced_stop_len)
            self.args = args
            prefix = f"datasets/tfbind8"

            # Read from file
            print(f"Loading data ...")
            with open(f"{prefix}/tfbind8-exact-v0-all.pkl", "rb") as f:
                oracle_d = pickle.load(f)

            munge = lambda x: "".join([str(c) for c in list(x)])
            self.oracle = {munge(x): float(y) for x, y in zip(oracle_d["x"], oracle_d["y"])}

            # Scale rewards
            all_rewards = np.array(list(self.oracle.values()))
            scaled_rewards = scale_rewards(
                all_rewards, args.beta, args.scale_reward_min, args.scale_reward_max
            )
            assert min(scaled_rewards) > 0
            self.scaled_oracle = {x: y for x, y in zip(self.oracle.keys(), scaled_rewards)}

            # Modes
            if args.mode_metric == "default":
                with open(f"{prefix}/modes_tfbind8.pkl", "rb") as f:
                    modes = pickle.load(f)
                self.modes = set([munge(x) for x in modes])
            elif args.mode_metric == "threshold":
                filename = f"{prefix}/modes_percentile_{args.mode_percentile}.pkl"
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        self.modes = pickle.load(f)
                else:
                    mode_percentile = args.mode_percentile
                    self.mode_r_threshold = np.percentile(
                        scaled_rewards, 100 * (1 - mode_percentile)
                    )
                    num_modes = int(len(self.scaled_oracle) * mode_percentile)
                    sorted_xs = sorted(self.scaled_oracle, key=self.scaled_oracle.get)
                    self.modes = set([x for x in sorted_xs[-num_modes:]])
                    with open(filename, "wb") as f:
                        pickle.dump(self.modes, f)
            elif args.mode_metric == "hammingball":
                filename = f"{prefix}/modes_hammingball_dist{args.mode_hammingball_dist}_percentile_{args.mode_percentile}.pkl"
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        self.modes = pickle.load(f)
                else:
                    mode_percentile = args.mode_percentile
                    self.mode_r_threshold = np.percentile(
                        scaled_rewards, 100 * (1 - mode_percentile)
                    )
                    self.mode_hammingball_dist = args.mode_hammingball_dist

                    print(
                        f"Computing modes with hamming ball distance {self.mode_hammingball_dist}..."
                    )
                    self.modes = set()
                    for x, y in tqdm(self.scaled_oracle.items()):
                        if y >= self.mode_r_threshold:
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

            print(f"Mode metric: {args.mode_metric}\tFound num modes: {len(self.modes)}")

            # compute true expected reward and logz
            log_rewards = np.log(scaled_rewards)
            self.logZ = torch.logsumexp(torch.from_numpy(log_rewards), dim=0).item()
            log_rewards_square = torch.logsumexp(torch.from_numpy(log_rewards * 2), dim=0).item()
            self.expected_reward = np.exp(log_rewards_square - self.logZ)

            print(
                f"Beta: {args.beta}\tExpected reward: {self.expected_reward:.2f}\tLogZ: {self.logZ:.2f}"
            )

            all_samples = list(self.oracle.keys())
            true_dist = np.exp(log_rewards - self.logZ)
            true_samples_idx = np.random.choice(
                len(self.oracle), size=args.eval_num_samples, p=true_dist, replace=True
            )
            self.true_samples = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx]

            del (
                all_rewards,
                scaled_rewards,
                log_rewards,
                log_rewards_square,
                all_samples,
                true_dist,
                true_samples_idx,
            )

        # Core
        def reward(self, x):
            assert x.is_leaf, "Error: Tried to compute reward on non-leaf node."
            return self.scaled_oracle[x.content]

        def is_mode(self, x, r):
            return x in self.modes

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
        trainer = Trainer(args, model, mdp, actor)
    elif args.model == "teacher":
        mdp_student = dynamic_inherit_mdp(base, args)
        mdp_teacher = dynamic_inherit_mdp(base, args)
        actor_student = actorclass(args, mdp_student)
        actor_teacher = actorclass(args, mdp_teacher)
        model_teacher, model_student = models.make_teacher_student_model(
            args, mdp_student, mdp_teacher, actor_student, actor_teacher
        )
        trainer = Trainer(args, model_student, mdp_student, teacher=model_teacher)
    else:
        mdp = dynamic_inherit_mdp(base, args)

        actor = actorclass(args, mdp)
        model = models.make_model(args, mdp, actor)
        trainer = Trainer(args, model, mdp)

    trainer.learn()
    return
