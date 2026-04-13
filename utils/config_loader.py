import tomli
from pathlib import Path

def load_config(config_path="config.toml"):
    """Loads the TOML configuration file."""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
        
    with open(config_path, "rb") as f:
        config = tomli.load(f)
    return config
