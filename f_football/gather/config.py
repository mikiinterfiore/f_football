import os

import sys
import requests
import yaml

import time
import datetime


class ConfigGatherer:

    def __init__(self, filepath):
        """_summary_

        Args:
            filepath (str): full filepath to the configuration file, with

        Returns:
            _type_: _description_
        """
        assert os.path.exists(filepath), ValueError("Specified path does not exists.")
        assert filepath.endswith(".yml") or filepath.endswith(".yaml"), ValueError("The accepted file needs to be yaml")

        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)

        self.data_sources = config['providers']
        self.database_dev = config['databases']['dev']
        self.database_prod = config['databases']['prod']


