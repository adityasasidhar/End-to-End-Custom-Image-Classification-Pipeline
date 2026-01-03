import yaml
import torch
import random
import numpy as np

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(cfg):
    if cfg["runtime"]["device"] == "cuda":
        return torch.device("cuda")
    if cfg["runtime"]["device"] == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
