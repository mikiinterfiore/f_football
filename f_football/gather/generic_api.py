import os
import sys
from typing import Union, Dict, List
import logging

import requests
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ApiGatherer:
    """
    General class to handle Api 
    """

    def __init__(self, key: dict, host: dict):
        """_summary_

        Args:
            key (str): _description_
            host (str): _description_
        """
        assert isinstance(key, dict)
        assert isinstance(host, dict)

        self.key_name = key['name']
        self.key_val = key['val']
        self.host_name = host['name']
        self.host_val = host['val']

    def __str__(self):
        return f"Api Interface for {self.host_name}"
    
    def __eq__(self, other):

        if isinstance(other, ApiGatherer):
            return self.host_val == other.host_val

    # def __hash__(self):
    #     return hash(self.host)

    # def __enter__(self):

    #     return self.dbconn

    # def __exit__(self, exc_type, exc_val, exc_tb):
        
    #     self.dbconn.close()
        

    def _send_request(self, url: str, req_headers: dict, req_params: dict):

        # get request
        logger.info(url)
        self.req: requests.Response = requests.get(url, headers = req_headers, params=req_params)

    
    def _verify_request(self):

        # safety checks
        status_code: int = self.req.status_code

        if status_code == requests.codes.ok:
            return req
        elif status_code == requests.codes.bad:
            sys.exit('Invalid request. Check parameters.')
        elif status_code == requests.codes.forbidden:
            sys.exit('This resource is restricted')
        elif status_code == requests.codes.not_found:
            sys.exit('This resource does not exist. Check parameters')
        elif status_code == requests.codes.too_many_requests:
            sys.exit('You have exceeded your allowed requests per minute/day')

    def _unpack_request(self):
        """_summary_

        Args:
            req (requests.Response): _description_
            search_type (str): _description_
            search_target (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.raw_data: dict = json.loads(self.req.text)
