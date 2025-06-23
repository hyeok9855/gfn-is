import argparse
import yaml
from argparse import Namespace


def parse_args() -> Namespace:
    """Control flow.
    Here:
    1. Read setting from sys.argv
    2. Load experiment-specific default settings in experiment folder.
    3. Use default yaml args to populate argparser.
      Read in CLI user --arguments with argparser and update options.
    In runexpwb.py:
    4. Log to wandb.config
    5. Use args = AttrDict(wandb.config) in code
    """
    setting_folds = {
        "bag": "exps/bag/",
        "tfbind8": "exps/tfbind8/",
        "tfbind10": "exps/tfbind10/",
        "qm9str": "exps/qm9str/",
        "sehstr": "exps/sehstr/",
        "gfp": "exps/gfp/",
        "utr": "exps/utr/",
        "rna": "exps/rna/",
    }
    # 1. Read setting from sys.argv
    argparser = argparse.ArgumentParser()
    argparser.add_argument(f"--setting", type=str)
    args_known, _ = argparser.parse_known_args()
    setting = args_known.setting

    # 2. Load experiment-specific default settings in experiment folder.
    config_yaml = setting_folds[setting] + "settings.yaml"
    print(f"Reading hyperparameters from {config_yaml} ...")
    with open(config_yaml) as f:
        default_args = yaml.load(f, Loader=yaml.FullLoader)

    # 2. Populate argparser and read user CLI args
    argparser = argparse.ArgumentParser()
    for arg, val in default_args.items():
        argparser.add_argument(f"--{arg}", default=val, type=type(val))
    parsed_args = Namespace(**vars(argparser.parse_args()))
    return parsed_args
