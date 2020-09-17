import os
import sys
import requests
import json

import random
import time
import pandas as pd

from utils_rapid_football import *


_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')

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
    while day_num_request < 100:
        for team in list(fixture_data.keys())[0:20]:
            # team = list(fixture_data.keys())[0]
            stats_fixt[team] = dict()
            for game in fixture_data[team]:
                # 30 requests max / minute
                time.sleep(random.randint(2, 4))
                fixture_focus = game['fixture_id']
                search_params = dict({'fixture_id' : fixture_focus})
                search_type = 'statistics_fixture'
                search_target = None
                stats_fixt[team][fixture_focus], day_num_request = wrap_api_request(day_num_request,
                                                                                        focus_source,
                                                                                        req_headers,
                                                                                        search_params,
                                                                                        search_type,
                                                                                        search_target)
    return None
