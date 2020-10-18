import os
import pickle
import json
import pandas as pd
import numpy as np

from sourcing.dnld_gaz import dnld_gaz_data
from sourcing.dnld_rapid import dnld_rapid_data

from manipulating.handle_league_calendar import get_updated_calendar
from manipulating.handle_fixtures import get_updated_fixture_master_stats
from manipulating.handle_players_features_target import create_pl_features_target
from manipulating.handle_team_features import create_tm_features

from modelling.predict_ffm import main as fv_model_predict


def ffm_main():

    # SETUP THE ARGUMENTS FOR THE FUNCTIONS
    focus_source = 'rapid_football'
    seasons = [2020]
    overwrite = True
    feat_windows = dict({
        'mean_wind' : [3,7],
        'sum_wind' : [3,7],
        'sd_wind' : [10]
    })

    # from the previous available calendar, find the last league round with data
    calendar_data = get_updated_calendar(focus_source)
    last_avail_round = _get_last_played_round(calendar_data)
    start_round = (last_avail_round - 2) if last_avail_round >= 3 else 1
    final_round = (last_avail_round + 2) if last_avail_round <= 36 else 39

    print('Scraping Gazzetta for Fantavoto and players stats')
    dnld_gaz_data(seasons, start_round, final_round)

    print('Calling %s API for league calendar and fixtures data.' % focus_source)
    dnld_rapid_data(focus_source, 2020, start_round, final_round, overwrite=True)
    print('Updating the league calendar.')
    calendar_data = get_updated_calendar(focus_source)
    print('Updating the fixture master and fixture stats data')
    get_updated_fixture_master_stats(focus_source)

    print('Computing the players features from Gazzetta data.')
    create_pl_features_target(feat_windows)
    print('Computing the team features from %s data.' % focus_source)
    create_tm_features(focus_source, feat_windows)

    print('Use the stored model to predict next game FV.')


    return None


def _get_last_played_round(calendar_data):

    seasons = calendar_data.groupby('league_id')['event_date'].min().reset_index()
    most_recent_season = seasons.loc[seasons['event_date'] == seasons['event_date'].max()]
    cal_idx = calendar_data['league_id'].isin(most_recent_season.loc[:, 'league_id'])
    match_status = calendar_data.loc[cal_idx,:].groupby('match')['statusShort'].value_counts()
    match_status = pd.DataFrame(match_status).rename(columns={'statusShort' : 'obs'}).reset_index()
    match_status['tot_match_obs'] = match_status.groupby('match')['obs'].transform('sum')
    match_status['match_type_pct'] = match_status['obs'] / match_status['tot_match_obs']
    finished_idx = match_status['statusShort'] == 'FT'
    relevant_idx = match_status['match_type_pct']>0.5
    last_avail_round = match_status.loc[(finished_idx) & (relevant_idx), 'match'].max()

    return last_avail_round

# if __name__ == "__main__":
#     ffm_main()
