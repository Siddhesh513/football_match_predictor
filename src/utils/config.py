import yaml
import os
from pathlib import Path


class Config:
    """Configuration manager for the project"""

    def __init__(self, config_path='config.yaml'):
        self.config_path = self._find_config(config_path)
        self.config = self._load_config()

    def _find_config(self, config_path):
        """Find config file in project structure"""
        # Try different paths
        paths_to_try = [
            config_path,
            Path(__file__).parent.parent.parent / config_path,
            Path.cwd() / config_path
        ]

        for path in paths_to_try:
            if Path(path).exists():
                return path

        raise FileNotFoundError(f"Config file not found: {config_path}")

    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        """Get configuration value by key (supports nested keys with dot notation)"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    @property
    def data_config(self):
        return self.config.get('data', {})

    @property
    def model_config(self):
        return self.config.get('model', {})

    @property
    def features_config(self):
        return self.config.get('features', {})
