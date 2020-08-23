import os
import sys

# import pandas as pd
# import numpy as np

import requests
import json


_BASE_DIR = '/home/costam/Documents'
_KEYS_DIR = os.path.join(_BASE_DIR, 'api_keys')
_PROVIDERS = dict({'football_data' : 'http://api.football-data.org/v2/',
                   'rapid_football' : 'http://something.something/'})

req_params = dict({})
req_params = dict({'areas' : '2114'})
focus_source = 'football_data'

def main(focus_source, req_params):
    api_key = get_api_creds(focus_source)
    req_params['X-Auth-Token'] = api_key
    # sending the request to the API
    req = send_request(focus_source, req_params)
    # extracting the data
    data = read_request_data(req)
    return None


def get_api_creds(focus_source):

    # check that the source is existing
    if focus_source not in list(_PROVIDERS.keys()):
        sys.exit("You can not have three process at the same time.")

    # selecting the right file
    all_files = os.listdir(_KEYS_DIR)
    provider = _PROVIDERS[focus_source]
    key_file = [x for x in all_files if focus_source in x][0]
    # Reading credentials from text
    with open(os.path.join(_KEYS_DIR, key_file), 'r') as f:
        api_key = f.readline().rstrip()
    return api_key


def send_request(focus_source, req_params):

    main_url = _PROVIDERS[focus_source]
    # working only with Serie A right now :
    # code = 'SA'
    # id = '2019'
    main_url = main_url+ 'competitions/SA/matches'
    req = requests.get(main_url, params=req_params)
    # safety checks
    status_code = req.status_code
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


def read_request_data(req):
    all_competitions = json.loads(req.text)
    # serie_a = [x for x in all_competitions['competitions'] if x['name']=='Serie A'][0]
    # serie_a['code']
