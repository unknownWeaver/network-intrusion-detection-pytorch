"""Loading utilities for the NSL-KDD dataset.

NSL-KDD ships as headerless CSVs. The 41 feature names below come from the
dataset's official documentation (KDD Cup 1999 task description, as carried
forward by NSL-KDD). Each row also has a `label` (attack name or "normal")
and a `difficulty_level` score assigned by the dataset creators based on how
many learners misclassified that record when the set was built - it's a
dataset-construction artifact, not a feature to train on.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty_level",
]

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]


def load_split(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    return pd.read_csv(path, names=COLUMN_NAMES)


def load_train() -> pd.DataFrame:
    return load_split("KDDTrain+.txt")


def load_test() -> pd.DataFrame:
    return load_split("KDDTest+.txt")
