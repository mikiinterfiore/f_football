import os
import sys
import requests
import json
import random
import time
import pandas as pd

_BASE_DIR = '/home/costam/Documents'
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')
_KEYS_DIR = os.path.join(_BASE_DIR, 'api_keys')
_PROVIDERS = dict({'football_data' : 'http://api.football-data.org/v2/',
                   'rapid_football' : 'https://api-football-v1.p.rapidapi.com/v2/'})


def get_api_creds(focus_source):

    # check that the source is existing
    if focus_source not in list(_PROVIDERS.keys()):
        sys.exit("The requested source API is not available. Please review.")

    # selecting the right file
    all_files = os.listdir(_KEYS_DIR)
    provider = _PROVIDERS[focus_source]
    key_file = [x for x in all_files if focus_source in x][0]
    # Reading credentials from text
    with open(os.path.join(_KEYS_DIR, key_file), 'r') as f:
        api_key = f.readline().rstrip()
    return api_key


def wrap_api_request(day_num_request, focus_source, req_headers, search_params, search_type, search_target=None):

    out_dir = os.path.join(_DATA_DIR, focus_source, search_type.replace('_', '').lower())
    # unpacking the search_params
    filename = ''
    for k in search_params:
        filename = filename + k.replace('_', '').lower() + '-' + str(search_params[k]) + '_'
    filename = filename + '.json'
    if filename not in os.listdir(out_dir):
        req = send_request(focus_source, search_params, search_type, req_headers)
        data = read_request_data(req, search_type, search_target)
        with open(os.path.join(out_dir, filename), 'w') as outfile:
            json.dump(data, outfile)
        day_num_request = day_num_request + 1
    else:
        with open(os.path.join(out_dir, filename), 'r') as inputfile:
            data = json.load(inputfile)
    return data, day_num_request


def build_request_url(focus_source, search_params, search_type):

    main_url = _PROVIDERS[focus_source]

    # CORE ELEMENTS TO USE :
    # country / code = where
    # season = when
    # league_id = unique torunament + year
    # team_id = unique
    # fixture_id = unique for every match
    # player_id = unique across career

    # Get fixtures by league_id and date
    # "fixtures/league/{league_id}/{date}?timezone=Europe/London"

    search_type_list = dict({
    'league_id' : 'leagues/country/%%code%%/%%season%%',
    'team_id' : 'teams/league/%%league_id%%',
    'fixture_id' : 'fixtures/team/%%team_id%%/%%league_id%%?timezone=Europe/London',
    # 'player_id' : 'players/fixture/%%fixture_id%%',
    'statistics_fixture' :'/statistics/fixture/%%fixture_id%%',
    'next_fixture' : 'fixtures/league/%%league_id%%/next/%%number%%?timezone=Europe/London',
    'league_round':'fixtures/league/%%league_id%%/Regular_Season_-_%%round%%',
    'player_id' : 'players/squad/%%team_id%%/%%season%%'
    })

    if search_type not in list(search_type_list.keys()):
        sys.exit('API calls for %s not currently available in the code.' % (search_type))

    url = main_url + search_type_list[search_type]
    for k in search_params.keys():
        target = '%%'+k+'%%'
        url = url.replace(target, str(search_params[k]))

    return url


def send_request(focus_source, search_params, search_type, req_headers):

    # construct the url with the parameters passsed
    url = build_request_url(focus_source, search_params, search_type)
    print(url)
    # get request
    req = requests.get(url, headers = req_headers)
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


def read_request_data(req, search_type, search_target = None):

    raw_data = json.loads(req.text)
    extract_focus = list(raw_data['api'].keys())[1]

    if search_type == 'statistics_fixture':
        # the structure of the data is different!!
        final_data = pd.DataFrame(columns=['fixture_stat', 'home', 'away'])
        for k in list(raw_data['api'][extract_focus].keys()):
            raw_dict = raw_data['api'][extract_focus][k]
            raw_dict['fixture_stat'] = k
            final_data = final_data.append(raw_dict, ignore_index=True)
        final_data = final_data.to_dict()

    elif search_type != 'statistics_fixture':
        final_data = []
        for d in raw_data['api'][extract_focus]:
            if search_target is None:
                # adding all the results from the call
                final_data.append(d)
            else:
                if d['name'] == search_target:
                    final_data.append(d)

    return final_data
