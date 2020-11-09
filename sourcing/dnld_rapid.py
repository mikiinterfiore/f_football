# RAPID FOOTBALL DATA API
# DETAILS FREE PLAN: 100/day requests MAX, 30/min requests MAX
# https://rapidapi.com/api-sports/api/api-football/details
# https://rapidapi.com/api-sports/api/api-football/tutorials/how-to-use-the-api-football-api-with-python

import os
import sys
import requests
import json

import random
import time
import pandas as pd

from utils.utils_rapid_football import *

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')


def dnld_rapid_data(focus_source, season, start_round, final_round, overwrite=False):

    # focus_source = 'rapid_football'
    # season = 2020
    # last_avail_round = 3
    # start_round = (last_avail_round - 2) if last_avail_round >= 3 else 1
    # final_round = (last_avail_round + 2) if last_avail_round <= 36 else 39
    # overwrite = True

    api_key = get_api_creds(focus_source)
    req_headers = dict({'X-RapidAPI-Key' : api_key,
                        'X-RapidAPI-host' : 'api-football-v1.p.rapidapi.com'})

    day_num_request = 0

    # get Serie A league_id
    # search_params = dict({'code' : 'IT', 'season' : 2018})
    # search_params = dict({'code' : 'IT', 'season' : 2019})
    search_params = dict({'code' : 'IT', 'season' : season})
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
                                                   search_type, search_target,
                                                   False)

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
                                                                       search_target,
                                                                       overwrite)
    print("API calls after completing team fixtures : %i" % day_num_request)

    # looking for the fixtures in every round
    # overwriting all the games in the last two rounds and all the games if __name__ == '__main__':
    # the next two rounds
    calendar = dict()
    for round in range(start_round,final_round):
        search_params = dict({'league_id' : seriea_id, 'round' : round})
        search_type = 'league_round'
        search_target = None
        # 30 requests max / minute
        time.sleep(random.randint(3, 4))
        if day_num_request >= 100:
            break
        calendar[round], day_num_request = wrap_api_request(day_num_request,
                                                            focus_source,
                                                            req_headers,
                                                            search_params,
                                                            search_type,
                                                            search_target,
                                                            overwrite)
    print("API calls after completing calendar : %i" % day_num_request)

    # get all statistics for a fixture_id

    # OLD CODE: looping through the teams data
    # for team in list(fixture_data.keys())[0:20]:
    #     # team = list(fixture_data.keys())[0]
    #     stats_fixt[team] = dict()
    #     for game in fixture_data[team]:

    stats_fixt = dict()
    for round in list(calendar.keys()):
        # round = list(calendar.keys())[0]
        stats_fixt[round] = dict()
        for game in calendar[round]:
            # 30 requests max / minute
            time.sleep(random.randint(3, 4))
            fixture_focus = game['fixture_id']
            search_params = dict({'fixture_id' : fixture_focus})
            search_type = 'statistics_fixture'
            search_target = None
            if day_num_request >= 100:
                break
            stats_fixt[round][fixture_focus], day_num_request = wrap_api_request(day_num_request,
                                                                                focus_source,
                                                                                req_headers,
                                                                                search_params,
                                                                                search_type,
                                                                                search_target,
                                                                                overwrite)

    print("API calls after completing fixtures stats data: %i" % day_num_request)

    return None
