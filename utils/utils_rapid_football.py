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


## API FUNCTIONS ---------------------------------------------------------------

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


def wrap_api_request(day_num_request, focus_source, req_headers, search_params,
                     search_type, search_target=None, overwrite=False):

    out_dir = os.path.join(_DATA_DIR, focus_source, search_type.replace('_', '').lower())
    # unpacking the search_params
    filename = ''
    for k in search_params:
        filename = filename + k.replace('_', '').lower() + '-' + str(search_params[k]) + '_'
    filename = filename + '.json'
    if (filename not in os.listdir(out_dir)) | (overwrite):
        req = _send_request(focus_source, search_params, search_type, req_headers)
        data = _read_request_data(req, search_type, search_target)
        if data is not None:
            with open(os.path.join(out_dir, filename), 'w') as outfile:
                json.dump(data, outfile)
        day_num_request = day_num_request + 1
    else:
        with open(os.path.join(out_dir, filename), 'r') as inputfile:
            data = json.load(inputfile)

    return data, day_num_request


def _build_request_url(focus_source, search_params, search_type):

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


def _send_request(focus_source, search_params, search_type, req_headers):

    # construct the url with the parameters passsed
    url = _build_request_url(focus_source, search_params, search_type)
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


def _read_request_data(req, search_type, search_target = None):

    raw_data = json.loads(req.text)
    extract_focus = list(raw_data['api'].keys())[1]

    final_data = None
    if raw_data['api']['results'] > 0:

        if search_type == 'statistics_fixture':
            # the structure of the data is different!!
            final_data = pd.DataFrame(columns=['fixture_stat', 'home', 'away'])
            # print(raw_data['api'][extract_focus].keys())
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

### DATA CLEANING FUNCTIONS ----------------------------------------------------

def get_fixture_stats_data(focus_source, fixtures_stats_file):

    if not os.path.isfile(fixtures_stats_file):
        fix_stats_cols = ['fixture_id', 'fixture_stat', 'home', 'away']
        fixtures_stats = pd.DataFrame(columns=fix_stats_cols)
        fixtures_stats.to_csv(fixtures_stats_file, header=True, index=False)

    fixtures_stats = pd.read_csv(fixtures_stats_file, header=0, index_col=False)

    return fixtures_stats


def update_fixtures_stats(fixtures_stats, fixtures_master, fixtures_stats_file, focus_source):

    for sf in fixtures_master['fixture_id'].unique():
        sf_file = os.path.join(_DATA_DIR, focus_source, 'statisticsfixture',
                                'fixtureid-'+str(sf)+'_.json')
        if os.path.isfile(sf_file):
            sf_data = pd.read_json(sf_file, orient = 'columns')
            sf_data['fixture_id'] = sf
            fixtures_stats = fixtures_stats.append(sf_data)

    return fixtures_stats.drop_duplicates()


def add_fixture_data(fixtures_master, sf):

    new_row = {
    'fixture_id' : int(sf['fixture_id']),
    'league_id' : int(sf['league_id']),
    # 'event_date' : dt.datetime.strptime(sf['event_date'],'%Y-%m-%dT%H:%M:S+%z'),
    'event_date' : pd.to_datetime(sf['event_timestamp'], unit='s'),
    # 'referee' : sf['referee'],
    'statusShort' : sf['statusShort'],
    'home_team_id' : int(sf['homeTeam']['team_id']),
    'home_team_name' : sf['homeTeam']['team_name'],
    'away_team_id' : int(sf['awayTeam']['team_id']),
    'away_team_name' : sf['awayTeam']['team_name']
    }

    if sf['status'] == 'Match Finished':
        if sf['score']['halftime'] is None:
            print(sf['fixture_id'])
            sf['score']['halftime'] = '0-0'
        new_row['home_ht_goal'] = int(sf['score']['halftime'].split('-')[0])
        new_row['away_ht_goal'] = int(sf['score']['halftime'].split('-')[1])
        new_row['home_ft_goal'] = int(sf['goalsHomeTeam'])
        new_row['away_ft_goal'] = int(sf['goalsAwayTeam'])
    else:
        # print(sf['status'])
        new_row['home_ht_goal'] = None
        new_row['away_ht_goal'] = None
        new_row['home_ft_goal'] = None
        new_row['away_ft_goal'] = None

    fixtures_master = fixtures_master.append(new_row, ignore_index=True)

    return fixtures_master


def get_fixture_master(focus_source, master_file):

    # creating the file master if not available
    if not os.path.isfile(master_file):
        fix_master_cols = ['fixture_id', 'league_id', 'event_date', #'referee',
                           'statusShort',
                           'home_team_id', 'away_team_id', 'home_team_name',
                           'away_team_name', 'home_ht_goal', 'away_ht_goal',
                           'home_ft_goal', 'away_ft_goal']
        fixtures_master = pd.DataFrame(columns=fix_master_cols)
        fixtures_master.to_csv(master_file, header=True, index=False)

    fixtures_master = pd.read_csv(master_file, header=0, index_col=False)
    return fixtures_master


def update_fixture_master(fixtures_master, fixture_dir):

    all_team_fixtures = os.listdir(fixture_dir)
    all_team_fixtures = [x for x in all_team_fixtures if x.endswith('.json')]
    for tf in all_team_fixtures:
        with open(os.path.join(fixture_dir, tf), 'r') as inputfile:
            raw_data = json.load(inputfile)
        for sf in raw_data:
            fixtures_master = add_fixture_data(fixtures_master, sf)

    # correctly setting the date and time
    fixtures_master['event_date'] = pd.to_datetime(fixtures_master['event_date'], utc=True)
    fixtures_master = fixtures_master.set_index('event_date', drop=True).tz_convert('Europe/London').reset_index()
    # print(fixtures_master.loc[fixtures_master['referee'].isna()])
    keep_idx = fixtures_master['statusShort'] == 'FT'
    fixtures_master = fixtures_master.loc[keep_idx].drop_duplicates()

    return fixtures_master
