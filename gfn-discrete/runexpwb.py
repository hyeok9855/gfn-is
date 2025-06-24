"""
Run experiment with wandb logging.

Usage:
python runexpwb.py --setting bag

Note: wandb isn't compatible with running scripts in subdirs:
  e.g., python -m exps.chess.chessgfn
So we call wandb init here.
"""

import random
import torch
import wandb
import numpy as np
from argparse import Namespace

from exps.tfbind8 import tfbind8
from exps.qm9str import qm9str
from exps.sehstr import sehstr
from exps.rna import rna
from options import parse_args


setting_calls = {
    "tfbind8": lambda args: tfbind8.main(args),
    "qm9str": lambda args: qm9str.main(args),
    "sehstr": lambda args: sehstr.main(args),
    "rna": lambda args: rna.main(args),
}


def main(args: Namespace) -> None:
    print(f"Using {args.setting} ...")
    exp_f = setting_calls[args.setting]
    exp_f(args)


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if args.setting == "rna":
        args.saved_models_dir = f"{args.saved_models_dir}/L{args.rna_length}_RNA{args.rna_task}/"
        wandb.init(
            project=f"{args.wandb_project}-L{args.rna_length}-{args.rna_task}",
            config=vars(args),
            mode=args.wandb_mode,
        )
    else:
        wandb.init(project=args.wandb_project, config=vars(args), mode=args.wandb_mode)

    run_name = args.model
    if args.model == "subtb":
        run_name += f"{args.lamda}"

    if args.num_offline_batches_per_round > 0:
        run_name += "_" + args.offline_select

    run_name += "_" + f"epsilon{args.explore_epsilon}"

    if args.sa_or_ssr == "ssr":
        run_name += "_" + args.sa_or_ssr

    if args.ls:
        run_name += "_" + "ls"
        if args.deterministic:
            run_name += "_" + "deterministic"
        run_name += "_" + f"k{args.k}"
        run_name += "_" + f"i{args.i}"

    run_name += "_" + f"beta{args.beta}"
    run_name += "_" + f"seed{args.seed}"
    run_name += "_" + f"buffer_size{args.replay_buffer_size}"
    run_name += "_" + f"alpha{args.alpha}"

    args.run_name = run_name.upper()
    print(f"Save model into {args.run_name}")

    args.device = args.device if torch.cuda.is_available() else "cpu"

    main(args)
