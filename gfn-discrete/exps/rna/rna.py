"""
RNA
from flexs
Start from scratch
"""

import os
import time
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from polyleven import levenshtein

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqarmdp

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

            print(f"Loading oracle ...")
            problem = flexs.landscapes.rna.registry()[f"L{self.rna_length}_RNA{self.rna_task}"]
            self.oracle = flexs.landscapes.RNABinding(**problem["params"])
            print(problem)

            self.SCALE_REWARD_MIN = args.scale_reward_min
            self.SCALE_REWARD_MAX = args.scale_reward_max
            self.REWARD_EXP = args.beta

            prefix = f"datasets/rna/L{self.rna_length}_RNA{self.rna_task}"
            allpreds_file = f"{prefix}_allpreds.pkl"

            if args.mode_metric == "threshold":
                filename = f"{prefix}_modes_percentile_{args.mode_percentile}_beta{args.beta}.pkl"
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        self.modes = pickle.load(f)
                else:
                    print("Loading data...")
                    start_time = time.time()
                    with open(allpreds_file, "rb") as f:
                        self.oracle_d = pickle.load(f)
                    py = np.array(list(self.oracle_d.values()))
                    print(f"Completed\t {time.time() - start_time:.2f}s")

                    mode_percentile = args.mode_percentile
                    print(f"Computing modes with threshold {mode_percentile}...")
                    self.mode_r_threshold = np.percentile(py, 100 * (1 - mode_percentile))
                    self.modes = set(
                        [x for x, y in tqdm(self.oracle_d.items()) if y >= self.mode_r_threshold]
                    )

                    with open(filename, "wb") as f:
                        pickle.dump(self.modes, f)

            elif args.mode_metric == "hammingball":
                filename = f"{prefix}_modes_hammingball_dist{args.mode_hammingball_dist}_percentile_{args.mode_percentile}_beta{args.beta}.pkl"
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        self.modes = pickle.load(f)
                else:
                    print("Loading data...")
                    start_time = time.time()
                    with open(allpreds_file, "rb") as f:
                        self.oracle_d = pickle.load(f)
                    py = np.array(list(self.oracle_d.values()))
                    print(f"Completed\t {time.time() - start_time:.2f}s")

                    mode_percentile = args.mode_percentile
                    self.mode_r_threshold = np.percentile(py, 100 * (1 - mode_percentile))
                    self.mode_hammingball_dist = args.mode_hammingball_dist

                    print(
                        f"Computing modes with hamming ball distance {self.mode_hammingball_dist}..."
                    )
                    self.modes = set()
                    for x, y in tqdm(self.oracle_d.items()):
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
            df = pd.read_csv(f"{prefix}_info.csv")
            if self.REWARD_EXP in df["reward_exp"].values.tolist():
                row = df[df["reward_exp"] == self.REWARD_EXP]
                self.logZ = row["logz"].values[0]
                self.expected_reward = row["expected_reward"].values[0]
                self.logZ_scaled = row["logz_scaled"].values[0]
                self.expected_reward_scaled = row["expected_reward_scaled"].values[0]
                self.scale = row["scale"].values[0]
            else:
                print("Loading data...")
                start_time = time.time()
                with open(allpreds_file, "rb") as f:
                    self.oracle_d = pickle.load(f)
                py = np.array(list(self.oracle_d.values()))
                print(f"Completed\t {time.time() - start_time:.2f}s")

                py = np.maximum(py, self.SCALE_REWARD_MIN)
                log_py = np.log(py)

                self.logZ = torch.logsumexp(torch.from_numpy(log_py * self.REWARD_EXP), 0).item()
                logr_square = torch.logsumexp(
                    torch.from_numpy(log_py * (self.REWARD_EXP + 1)), 0
                ).item()
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

                self.scale = np.exp(log_scale)

                del py
                del log_py
                del self.oracle_d

            print(
                f"Beta: {self.REWARD_EXP}\tExpected reward: {self.expected_reward:.2f}\tLogZ: {self.logZ:.2f}"
            )

            true_sample_file = f"{prefix}_true_samples_beta{args.beta}.pkl"
            if os.path.exists(true_sample_file):
                with open(true_sample_file, "rb") as f:
                    true_samples_info = pickle.load(f)
                self.true_samples = true_samples_info["true_samples"]
                self.true_samples_scaled = true_samples_info["true_samples_scaled"]
            else:
                print("Loading data...")
                start_time = time.time()
                with open(allpreds_file, "rb") as f:
                    self.oracle_d = pickle.load(f)
                py = np.array(list(self.oracle_d.values()))
                print(f"Completed\t {time.time() - start_time:.2f}s")

                py = np.maximum(py, self.SCALE_REWARD_MIN)
                log_py = np.log(py)

                log_py_scaled = log_py * self.REWARD_EXP
                log_scale = np.log(self.SCALE_REWARD_MAX) - max(log_py_scaled)
                log_py_scaled = log_py_scaled + log_scale

                print("Computing true samples...")
                all_samples = list(self.oracle_d.keys())
                # too many samples - probabilities do not sum to 1
                # true_dist = np.exp(log_py * self.REWARD_EXP - self.logZ)
                # true_samples_idx = np.random.choice(len(self.oracle_d), size=args.eval_num_samples, p=true_dist)

                # true_samples_idx = []
                # logprobs = log_py * self.REWARD_EXP - self.logZ
                # for _ in tqdm(range(self.args.eval_num_samples)):
                #   gumbels = np.random.gumbel(size=(len(self.oracle_d)))
                #   true_samples_idx.append(np.argmax(logprobs + gumbels))
                # self.true_samples = [self.state(all_samples[i], is_leaf=True) for i in true_samples_idx]

                true_samples_idx_scaled = []
                logprobs_scaled = log_py_scaled - self.logZ_scaled
                for _ in tqdm(range(self.args.eval_num_samples)):
                    gumbels = np.random.gumbel(size=(len(self.oracle_d)))
                    true_samples_idx_scaled.append(np.argmax(logprobs_scaled + gumbels))
                self.true_samples_scaled = [
                    self.state(all_samples[i], is_leaf=True) for i in true_samples_idx_scaled
                ]

                with open(true_sample_file, "wb") as f:
                    pickle.dump(
                        {
                            "true_samples": self.true_samples_scaled,
                            "true_samples_scaled": self.true_samples_scaled,
                        },
                        f,
                    )
                del py
                del log_py
                del self.oracle_d

        # Core
        def reward(self, x):
            assert x.is_leaf, "Error: Tried to compute reward on non-leaf node."
            r = self.oracle.get_fitness([x.content]).item()

            r = np.maximum(r, self.SCALE_REWARD_MIN)
            r = r**self.REWARD_EXP
            r = r * self.scale + 1e-12
            return r

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


def number_of_modes(args):
    print("Running evaluation RNA ...")

    # load model checkpoint
    ckpt_path = args.saved_models_dir + args.run_name
    with open(ckpt_path + "/" + f"final_sample.pkl", "rb") as f:
        generated_samples = pickle.load(f)

    mode_info_file = args.mode_info_file + f"L{args.rna_length}_RNA{args.rna_task}/mode_info.pkl"
    with open(mode_info_file, "rb") as f:
        mode_info = pickle.load(f)

    unique_samples = set()
    batch_size = args.num_samples_per_online_batch
    number_of_modes = {k: np.zeros((len(generated_samples) // batch_size,)) for k in mode_info}
    with tqdm(total=len(generated_samples)) as pbar:
        for i in range(0, len(generated_samples), batch_size):
            for exp in generated_samples[i : i + batch_size]:
                if exp.x not in unique_samples:
                    if exp.x.content in mode_info["modes"]:
                        number_of_modes["modes"][i // batch_size] += 1
                    if exp.x.content in mode_info["modes_hamming_ball1"]:
                        number_of_modes["modes_hamming_ball1"][i // batch_size] += 1
                    if exp.x.content in mode_info["modes_hamming_ball2"]:
                        number_of_modes["modes_hamming_ball2"][i // batch_size] += 1
                    unique_samples.add(exp.x)
            pbar.update(batch_size)
            pbar.set_postfix(number_of_modes=np.sum(number_of_modes["modes"]))
    print(np.sum(number_of_modes["modes"]))
    np.savez_compressed(
        ckpt_path + "/" + f"number_of_modes_updated.npz",
        modes=number_of_modes["modes"],
        modes_hamming_ball1=number_of_modes["modes_hamming_ball1"],
        modes_hamming_ball2=number_of_modes["modes_hamming_ball2"],
    )
