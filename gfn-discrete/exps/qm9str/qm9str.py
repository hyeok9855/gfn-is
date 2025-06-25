"""
qm9 as string
"""

import os
import pickle, functools
import numpy as np
from tqdm import tqdm
import torch
from argparse import Namespace

from gflownet.MDPs import molstrmdp
from gflownet.GFNs import models
from gflownet.trainers import Trainer
from gflownet.utils.misc_utils import scale_rewards

from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity


class QM9stringMDP(molstrmdp.MolStrMDP):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        prefix = f"datasets/qm9str"

        # Read from file
        print(f"Loading data ...")
        with open(f"{prefix}/block_qm9str_v1_s5.pkl", "rb") as f:
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
            filename = f"{prefix}/modes_percentile_{args.mode_percentile}_beta{args.beta}.pkl"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    self.modes = pickle.load(f)
            else:
                mode_percentile = args.mode_percentile
                self.mode_r_threshold = np.percentile(scaled_rewards, 100 * (1 - mode_percentile))
                num_modes = int(len(self.scaled_oracle) * mode_percentile)
                sorted_xs = sorted(self.scaled_oracle, key=self.scaled_oracle.get)
                self.modes = set(sorted_xs[-num_modes:])
                with open(f"{filename}", "wb") as f:
                    pickle.dump(self.modes, f)

        elif args.mode_metric == "tanimoto":
            filename = f"{prefix}/modes_tanimoto_div{args.mode_div_threshold}_percentile_{args.mode_percentile}_beta{args.beta}.pkl"
            if os.path.exists(filename):
                with open(filename, "rb") as f:
                    self.modes = pickle.load(f)
            else:
                mode_percentile = args.mode_percentile
                self.mode_r_threshold = np.percentile(scaled_rewards, 100 * (1 - mode_percentile))
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

        # generate samples from true distribution
        all_samples = list(self.oracle.keys())
        true_dist = np.exp(log_rewards - self.logZ)
        true_samples_idx = np.random.choice(
            len(self.oracle), size=args.eval_num_samples, p=true_dist
        )
        self.true_samples = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx]

    # Core
    def reward(self, x):
        assert x.is_leaf, "Error: Tried to compute reward on non-leaf node."
        return self.scaled_oracle[x.content]

    def is_mode(self, x, r):
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


def main(args: Namespace) -> None:
    print("Running experiment qm9str ...")

    if args.model == "gafn":
        mdp = QM9stringMDP(args)
        actor = molstrmdp.MolStrActor(args, mdp)
        rnd_target = molstrmdp.MolStrActor(args, mdp)
        rnd_predict = molstrmdp.MolStrActor(args, mdp)
        model = models.make_model(args, mdp, actor, rnd_target=rnd_target, rnd_predict=rnd_predict)
        trainer = Trainer(args, model, mdp)
    elif args.model == "teacher":
        mdp_student = QM9stringMDP(args)
        mdp_teacher = QM9stringMDP(args)
        actor_student = molstrmdp.MolStrActor(args, mdp_student)
        actor_teacher = molstrmdp.MolStrActor(args, mdp_teacher)
        model_teacher, model_student = models.make_teacher_student_model(
            args, mdp_student, mdp_teacher, actor_student, actor_teacher
        )
        trainer = Trainer(args, model_student, mdp_student, teacher=model_teacher)
    else:
        mdp = QM9stringMDP(args)
        actor = molstrmdp.MolStrActor(args, mdp)
        model = models.make_model(args, mdp, actor)
        trainer = Trainer(args, model, mdp)
    trainer.learn()
    return


def number_of_modes(args):
    print("Count number of modes qm9str ...")

    # load model checkpoint
    ckpt_path = args.saved_models_dir + args.run_name
    with open(ckpt_path + "/" + f"final_sample.pkl", "rb") as f:
        generated_samples = pickle.load(f)

    with open(args.mode_info_file, "rb") as f:
        mode_info = pickle.load(f)

    unique_samples = set()
    batch_size = args.num_samples_per_online_batch
    number_of_modes = {k: np.zeros((len(generated_samples) // batch_size,)) for k in mode_info}
    with tqdm(total=len(generated_samples)) as pbar:
        for i in range(0, len(generated_samples), batch_size):
            for exp in generated_samples[i : i + batch_size]:
                if exp.x not in unique_samples:
                    if exp.x.content in mode_info["modes_div_threshold_075"]:
                        number_of_modes["modes_div_threshold_075"][i // batch_size] += 1
                    if exp.x.content in mode_info["modes_div_threshold_05"]:
                        number_of_modes["modes_div_threshold_05"][i // batch_size] += 1
                    if exp.x.content in mode_info["modes"]:
                        number_of_modes["modes"][i // batch_size] += 1
                unique_samples.add(exp.x)
            pbar.update(batch_size)
            pbar.set_postfix(number_of_modes=np.sum(number_of_modes["modes"]))
    print(np.sum(number_of_modes["modes"]))
    np.savez_compressed(
        ckpt_path + "/" + f"number_of_modes_updated.npz",
        modes=number_of_modes["modes"],
        modes_div_threshold_05=number_of_modes["modes_div_threshold_05"],
        modes_div_threshold_075=number_of_modes["modes_div_threshold_075"],
    )
