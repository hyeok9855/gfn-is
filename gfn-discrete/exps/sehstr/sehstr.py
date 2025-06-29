"""
seh as string
"""

import os
import pickle, functools
import numpy as np
from tqdm import tqdm
import torch

from gflownet.MDPs import molstrmdp
from gflownet.GFNs import models

# from datasets.sehstr import gbr_proxy
from gflownet.utils.misc_utils import scale_rewards
from gflownet.trainers import Trainer


from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity


class SEHstringMDP(molstrmdp.MolStrMDP):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        prefix = f"datasets/sehstr"

        # self.proxy_model = gbr_proxy.sEH_GBR_Proxy(args)

        with open(f"{prefix}/block_18_stop6.pkl", "rb") as f:
            self.oracle = pickle.load(f)

        # scale rewards
        all_rewards = np.array(list(self.oracle.values()))
        scaled_rewards = scale_rewards(
            all_rewards, args.beta, args.scale_reward_min, args.scale_reward_max
        )
        assert min(scaled_rewards) > 0
        self.scaled_oracle = {x: y for x, y in zip(self.oracle.keys(), scaled_rewards)}

        # define modes as top % of xhashes and diversity metrics
        if args.mode_metric == "threshold":
            filename = f"{prefix}/modes_percentile_{args.mode_percentile}.pkl"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    self.modes = pickle.load(f)
            else:
                mode_percentile = args.mode_percentile
                self.mode_r_threshold = np.percentile(scaled_rewards, 100 * (1 - mode_percentile))
                num_modes = int(len(self.scaled_oracle) * mode_percentile)
                sorted_xs = sorted(self.scaled_oracle, key=self.scaled_oracle.get)
                self.modes = set(sorted_xs[-num_modes:])
                with open(filename, "wb") as f:
                    pickle.dump(self.modes, f)
        elif args.mode_metric == "tanimoto":
            filename = f"{prefix}/modes_tanimoto_div{args.mode_div_threshold}_percentile_{args.mode_percentile}.pkl"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    self.modes = pickle.load(f)
            else:
                self.mode_r_threshold = np.percentile(
                    scaled_rewards, 100 * (1 - args.mode_percentile)
                )
                self.mode_div_threshold = args.mode_div_threshold

                print(
                    f"Computing modes using tanimoto similarity with threshold {self.mode_div_threshold}..."
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
                                diversity_score = self.dist_states(x, mode)
                                if diversity_score <= self.mode_div_threshold:
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

        true_samples_filename = f"{prefix}/samples_beta{args.beta}_n{args.eval_num_samples}.pkl"
        if os.path.exists(true_samples_filename):
            with open(true_samples_filename, "rb") as f:
                self.true_samples = pickle.load(f)
        else:
            # generate samples from true distribution
            all_samples = list(self.oracle.keys())
            true_dist = np.exp(log_rewards - self.logZ)
            true_samples_idx = np.random.choice(
                len(self.oracle), size=args.eval_num_samples, p=true_dist, replace=True
            )
            self.true_samples = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx]

            with open(true_samples_filename, "wb") as f:
                pickle.dump(self.true_samples, f)
            del all_samples, true_dist, true_samples_idx
        del all_rewards, scaled_rewards, log_rewards, log_rewards_square

    # Core
    @functools.lru_cache(maxsize=None)
    def reward(self, x):
        assert x.is_leaf, "Error: Tried to compute reward on non-leaf node."
        return self.scaled_oracle[x.content]

    def is_mode(self, x, r):
        if self.args.mode_metric == "threshold":
            return r >= self.mode_r_threshold
        else:
            return x.content in self.modes

    # Diversity
    def dist_states(self, state1, state2):
        """Tanimoto similarity on morgan fingerprints"""
        fp1 = self.get_morgan_fp(state1)
        fp2 = self.get_morgan_fp(state2)
        return 1 - FingerprintSimilarity(fp1, fp2)

    @functools.lru_cache(maxsize=None)
    def get_morgan_fp(self, state):
        mol = self.state_to_mol(state)
        fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        return fp


def main(args):
    print("Running experiment sehstr ...")
    if args.model == "gafn":
        mdp = SEHstringMDP(args)
        actor = molstrmdp.MolStrActor(args, mdp)
        rnd_target = molstrmdp.MolStrActor(args, mdp)
        rnd_predict = molstrmdp.MolStrActor(args, mdp)
        model = models.make_model(args, mdp, actor, rnd_target=rnd_target, rnd_predict=rnd_predict)
        trainer = Trainer(args, model, mdp)
    elif args.model == "teacher":
        mdp_student = SEHstringMDP(args)
        mdp_teacher = SEHstringMDP(args)

        actor_student = molstrmdp.MolStrActor(args, mdp_student)
        actor_teacher = molstrmdp.MolStrActor(args, mdp_teacher)

        model_teacher, model_student = models.make_teacher_student_model(
            args, mdp_student, mdp_teacher, actor_student, actor_teacher
        )
        trainer = Trainer(args, model_student, mdp_student, teacher=model_teacher)
    else:
        mdp = SEHstringMDP(args)
        actor = molstrmdp.MolStrActor(args, mdp)
        model = models.make_model(args, mdp, actor)
        trainer = Trainer(args, model, mdp)

    trainer.learn()
    return
