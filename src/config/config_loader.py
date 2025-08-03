# src/config/config_loader.py

import yaml

def load_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)