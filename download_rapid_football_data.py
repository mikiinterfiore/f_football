import os
import sys
import requests
import json


_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')
_KEYS_DIR = os.path.join(_BASE_DIR, 'api_keys')
_PROVIDERS = dict({'football_data' : 'http://api.football-data.org/v2/',
                   'rapid_football' : 'https://api-football-v1.p.rapidapi.com/v2/'})

focus_source = 'rapid_football'


def main(focus_source, req_params):

    api_key = get_api_creds(focus_source)
    req_headers = dict({'X-RapidAPI-Key' : api_key,
                        'X-RapidAPI-host' : 'api-football-v1.p.rapidapi.com'})

    # get Serie A league_id
    search_params = dict({'code' : 'IT', 'season' : 2019})
    search_type='league_id'
    search_target='Serie A'
    seriea_data = wrap_api_request(focus_source, req_headers, search_params,
                                   search_type, search_target)
    seriea_id = seriea_data[0]['league_id']

    # get all team_id (for a league_id)
    search_params = dict({'league_id' : seriea_id})
    search_type = 'league_id'
    search_target = None
    teams_data = wrap_api_request(focus_source, req_headers, search_params,
                                   search_type, search_target)
    # get all fixture_id (for a league_id)

    # get all statistics for a fixture_id

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


def wrap_api_request(focus_source, req_headers, search_params, search_type, search_target):

    filename = focus_source + '_' + search_target.replace(' ', '').lower() + '_' + str(search_params['season']) + '.json'
    if filename not in os.listdir(_DATA_DIR):
        seriea_req = send_request(focus_source, search_params,
                                  search_type='league_id')
        seriea_data = read_request_data(req = seriea_req, search_type='league_id', search_target='Serie A')
        # seriea_data = final_data
        with open(os.path.join(_DATA_DIR, filename), 'w') as outfile:
            json.dump(seriea_data, outfile)
    else:
        with open(os.path.join(_DATA_DIR, filename), 'r') as inputfile:
            seriea_data = json.load(inputfile)
    return data


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
    'fixture_id' : 'fixtures/team/%%team_id%%/%%league_id%%?timezone=%%timezone%%"',
    'player_id' : 'players/fixture/%%fixture_id%%'
    })

    if search_type not in list(search_type_list.keys()):
        sys.exit('API calls for %s not currently available in the code.' % (search_type))

    url = main_url + search_type_list[search_type]
    for k in search_params.keys():
        target = '%%'+k+'%%'
        url = final_url.replace(target, str(search_params[k]))

    return url


def send_request(focus_source, search_params, search_type):

    # construct the url with the parameters passsed
    url = build_request_url(focus_source, search_params, search_type)
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

    final_data = []

    for d in raw_data['api'][extract_focus]:
        print(d[search_type])
        # todo(Michele): verify that this "name" convention is valid across search_targets
        if search_target is None:
            # adding all the results from the call
            final_data.append(d)
        else:
            if d['name'] == search_target:
                final_data.append(d)

    return final_data
