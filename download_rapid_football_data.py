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

# focus_source = 'rapid_football'
# DETAILS FREE PLAN: 100/day requests MAX, 30/min requests MAX
# https://rapidapi.com/api-sports/api/api-football/details
# https://rapidapi.com/api-sports/api/api-football/tutorials/how-to-use-the-api-football-api-with-python

def main(focus_source):

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

    # get all team_id (for a league_id)
    search_params = dict({'league_id' : seriea_id})
    search_type = 'team_id'
    search_target = None
    teams_data, day_num_request = wrap_api_request(day_num_request, focus_source,
                                                   req_headers, search_params,
                                                   search_type, search_target)

    # get all fixture_id (for a league_id)
    fixture_data = dict()
    for t in teams_data:
        search_params = dict({'league_id' : seriea_id, 'team_id' : t['team_id']})
        search_type = 'fixture_id'
        search_target = None
        fixture_data[t['team_id']], day_num_request = wrap_api_request(day_num_request,
                                                                       focus_source,
                                                                       req_headers,
                                                                       search_params,
                                                                       search_type,
                                                                       search_target)
    # get all statistics for a fixture_id
    stats_fixt = dict()
    for team in list(fixture_data.keys())[0:20]:
        # team = list(fixture_data.keys())[0]
        stats_fixt[team] = dict()
        for game in fixture_data[team]:
            time.sleep(random.randint(5, 8))
            fixture_focus = game['fixture_id']
            search_params = dict({'fixture_id' : fixture_focus})
            search_type = 'statistics_fixture'
            search_target = None
            if day_num_request < 100:
                stats_fixt[team][fixture_focus], day_num_request = wrap_api_request(day_num_request,
                                                                                    focus_source,
                                                                                    req_headers,
                                                                                    search_params,
                                                                                    search_type,
                                                                                    search_target)
    return None


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
        req = send_request(focus_source, search_params, search_type)
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

    search_type_list = dict({
    'league_id' : 'leagues/country/%%code%%/%%season%%',
    'team_id' : 'teams/league/%%league_id%%',
    'fixture_id' : 'fixtures/team/%%team_id%%/%%league_id%%?timezone=Europe/London',
    'player_id' : 'players/fixture/%%fixture_id%%',
    'statistics_fixture' :'/statistics/fixture/%%fixture_id%%'
    })

    if search_type not in list(search_type_list.keys()):
        sys.exit('API calls for %s not currently available in the code.' % (search_type))

    url = main_url + search_type_list[search_type]
    for k in search_params.keys():
        target = '%%'+k+'%%'
        url = url.replace(target, str(search_params[k]))

    return url


def send_request(focus_source, search_params, search_type):

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
