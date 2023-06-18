from typing import Union, Dict, List
import logging
import json

import random
import time
import datetime

import pandas as pd

from f_football.gather.generic_api import ApiGatherer
from f_football.utils.utils_rapid_football import *

logger = logging.getLogger(__name__)


class RapidFootGatherer(ApiGatherer):

    def __init__(self, key, host):
            
            super().__init__(key, host)

            self.headers = {
                    self.key_name: self.key_val,
                    self.host_name: self.host_val
            }
            
            # endpoints
            # self.api_search_patterns = {
            #     'league_id' : 'v3/leagues/country/%%code%%/%%season%%',
            #     'team_id' : 'v3/teams/league/%%league_id%%',
            #     'fixture_id' : 'v3/fixtures/team/%%team_id%%/%%league_id%%?timezone=Europe/London',
            #     'statistics_fixture' :'v3/statistics/fixture/%%fixture_id%%',
            #     'next_fixture' : 'v3/fixtures/league/%%league_id%%/next/%%number%%?timezone=Europe/London',
            #     'league_round':'v3/fixtures/league/%%league_id%%/Regular_Season_-_%%round%%',
            #     'player_id' : 'v3/players/squad/%%team_id%%/%%season%%'
            # }

            self.endpoints = {
                'league_id': 'leagues',
                'team_id': 'teams',
                'fixture_id': 'fixtures',
                'statistics_fixture': 'fixtures/statistics',
                'next_fixture': 'fixtures',
                'league_round': 'fixtures',
                'player_id': 'players'
            }

            self.query_params = {
                'league_id' : {"code": None}, # by country code 
                'team_id' : {'league': None, 'season': None},
                'fixture_id' : {'season': None, "team": None, "timezone": "Europe/London"},
                'statistics_fixture' : {"fixture": None},
                'next_fixture' : {'league': None, 'next': None, "timezone": "Europe/London"},
                'league_round': {"league": None,"season":None, "round": None}, # "Regular Season - 10"
                'player_id' : {"team": None,"season": None}
            }

            self.field_focus = {
                'league_id' : 'league',
                'team_id' : 'team',
                'fixture_id' : 'fixture',
                'statistics_fixture' : None,
                'next_fixture' : None,
                'league_round': 'fixture',
                'player_id' : ''
            }

            # TODO add referees, coach, injuries, transfers, 

    def build_url(self, search_item: str):
        """_summary_

        Args:
            focus_source (str): _description_
            search_params (dict): _description_
            search_type (str): _description_

        Returns:
            _type_: _description_
        """

        # CORE ELEMENTS TO USE :
        # country / code = where
        # season = when
        # league_id = unique torunament + year
        # team_id = unique
        # fixture_id = unique for every match
        # player_id = unique across career

        # Get fixtures by league_id and date
        # "fixtures/league/{league_id}/{date}?timezone=Europe/London"

        assert search_item in list(self.endpoints.keys()), f'API calls for {search_item} not currently available in the code.'
        
        # basic building block for API get
        url = 'https://' + self.host_val + '/v3/' + self.endpoints[search_item]

        # TODO improve this replacement strategy

        # # adding the specific target items
        # for k, v in search_params.items():
        #     url = url.replace('%%'+k+'%%', str(v))
            
        return url

    def _extract_data(self, search_item: str) -> Union[pd.DataFrame, List]:
        """_summary_

        Args:
            req (requests.Response): _description_
            search_type (str): _description_
            search_target (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        self.final_data = None
        
        # extract_focus = list(self.raw_data['response'].keys())[1]

        if len(self.raw_data['response']) > 0:

            if search_item in ('fixture_id', 'league_round'):

                # TODO unpack nested dictionaries for fixture
                pass
            
            elif search_item == 'statistics_fixture':
                
                # final_data = pd.DataFrame(columns=['fixture_stat', 'home', 'away'])
                # for k in list(self.raw_data['response'][extract_focus].keys()):
                #     raw_dict = self.raw_data['response'][extract_focus][k]
                #     raw_dict['fixture_stat'] = k
                #     final_data = final_data.append(raw_dict, ignore_index=True)
                
                # TODO check whether using long or wide dataframe structure
                final_data = pd.concat([ 
                    pd.DataFrame(x['statistics']).assign(team_id=x['team']['id']) for x in self.raw_data['response']
                ], axis=0).assign(fixture_id=self.raw_data['parameters']['fixture'])

                self.final_data = final_data.to_dict(orient='records')

            else:

                self.final_data = pd.DataFrame([
                    x[self.field_focus[search_item]] for x in self.raw_data['response']
                    ]).to_dict(orient='records')

                # for d in [extract_focus]:
                #     if search_target is None:
                #         # adding all the results from the call
                #         self.final_data.append(d)
                #     else:
                #         if d['name'] == search_target:
                #             self.final_data.append(d)

    def wrap_api_request(self, search_item, search_params, day_num_request, focus_source, search_target=None, overwrite=False):
        """

        Args:
            day_num_request (_type_): _description_
            focus_source (_type_): _description_
            req_headers (_type_): _description_
            search_params (_type_): _description_
            search_type (_type_): _description_
            search_target (_type_, optional): _description_. Defaults to None.
            overwrite (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """

        # out_dir = os.path.join(_DATA_DIR, focus_source, search_type.replace('_', '').lower())
        
        # # unpacking the search_params
        # filename = ''
        # for k in search_params:
        #     filename = filename + k.replace('_', '').lower() + '-' + str(search_params[k]) + '_'
        # filename = filename + '.json'

        # if (filename not in os.listdir(out_dir)) | (overwrite):

        url = self.build_url(search_item=search_item, search_params=search_params)
        
        self._send_request(url, self.headers)
        self._unpack_request()
        self._extract_data(req, search_type, search_target)

        if data is not None:
            with open(os.path.join(out_dir, filename), 'w') as outfile:
                json.dump(data, outfile)
            day_num_request = day_num_request + 1
        else:
            with open(os.path.join(out_dir, filename), 'r') as inputfile:
                data = json.load(inputfile)

        return data, day_num_request


