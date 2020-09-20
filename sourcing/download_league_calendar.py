import os
import sys
import requests
import json

import random
import time
import pandas as pd

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')
_KEYS_DIR = os.path.join(_BASE_DIR, 'api_keys')
_PROVIDERS = dict({'football_data' : 'http://api.football-data.org/v2/',
                   'rapid_football' : 'https://api-football-v1.p.rapidapi.com/v2/'})


def main(focus_source = 'rapid_football'):

    api_key = get_api_creds(focus_source)
    req_headers = dict({'X-RapidAPI-Key' : api_key,
                        'X-RapidAPI-host' : 'api-football-v1.p.rapidapi.com'})

    day_num_request = 0
    # while day_num_request < 100:
    # get Serie A league_id
    search_params = dict({'code' : 'IT', 'season' : 2019})
    search_type='league_id'
    search_target='Serie A'
    seriea_data, day_num_request = wrap_api_request(day_num_request, focus_source,
                                                    req_headers, search_params,
                                                    search_type, search_target)
    seriea_id = seriea_data[0]['league_id']

    # looking at the upcoming matches
    search_params = dict({'league_id' : seriea_id, 'number' : 10})
    search_type = 'next_fixture'
    search_target = None
    next_fixtures, day_num_request = wrap_api_request(day_num_request,
                                                      focus_source,
                                                      req_headers,
                                                      search_params,
                                                      search_type,
                                                      search_target)

    return None
