import os
import sys

# import pandas as pd
# import numpy as np

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
    # sending the request to the API
    req = send_request(focus_source, req_params, req_headers)
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


def build_request_url(focus_source, search_params):

    main_url = _PROVIDERS[focus_source]

    # working only with Serie A right now : each season has a different league_id
    # so we have to iterate if we want to go back in time

    # CORE ELEMENTS TO USE :
    # country / code = where
    # season = when
    # league_id = unique torunament + year
    # team_id = unique
    # fixture_id = unique for every match
    # player_id = unique across career

    search_params = dict({'code' : 'IT',
                          'country' : 'Italy',
                          'league_id' : ,
                          'season' : 2019,
                          'team_id' : ,
                          'end_date' : ,
                          'timezone' : 'Europe/London',
                          'number' : 3, # past / upcoming number of matches
                          'fixture_id' : , # single game id!
                          'player_id' : })

    # searching leagues
    'leagues/league/{%%league_id%%}'
    'leagues/seasonsAvailable/{%%league_id%%}'
    'leagues/team/{%%team_id%%}'
    'leagues/country/{%%country%%}'
    'leagues/country/{%%code%%}'
    'leagues/country/{%%code%%}/{%%season%%}'
    'leagues/country/{%%country%%}/{%%season%%}'
    # searching teams
    'teams/team/{%%team_id%%}'
    'teams/league/{%%league_id%%}'
    # searching stats
    'statistics/{%%league_id%%}/{%team_id%}'
    'statistics/{%%league_id%%}/{%team_id%}/{%%end_date%%}'
    # searching standings is not point in time, just latest available
    # searching fixtures :
    'fixtures/rounds/{%league_id%}'
    'fixtures/team/{%%team_id%%}/{%%league_id%%}?timezone=%%timezone%%"'
    'fixtures/team/{%%team_id%%}/next/{%%number%%}?timezone=%%timezone%%' # not restricted to league_id
    'fixtures/team/{%%team_id%%}/last/{%%number%%}?timezone=%%timezone%%' # not restricted to league_id
    # searching stats by fixtures : REVIEW!
    'statistics/fixture/{%%fixture_id%%}'
    # searcing events
    'events/fixture/{%%fixture_id%%}'
    # searching lineups
    'lineups/{%%fixture_id%%}'
    # searching players
    'players/seasons'
    'players/player/{%%player_id%%}'
    'players/player/{%%player_id%%}/{%%season%%}'
    'players/fixture/{%%fixture_id%%}'




    final_url = main_url


    return final_url



def send_request(focus_source, search_params):

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


def read_request_data(req):
    all_competitions = json.loads(req.text)
    # serie_a = [x for x in all_competitions['competitions'] if x['name']=='Serie A'][0]
    # serie_a['code']
