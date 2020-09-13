import os
# import sys
# import json
import pandas as pd
import numpy as np
# import datetime as dt
# from dateutil import tz

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main(focus_source = 'rapid_football'):

    # load the fixtures master
    master_file = os.path.join(_DATA_DIR, focus_source, 'masters', 'fixture_master.csv')
    fixtures_master = pd.read_csv(master_file, header=0, index_col=False)
    goals_stats = get_goals_fixture_stats(fixtures_master)
    fixtures_stats = get_detailed_fixture_stats(focus_source)
    # RAW DATA FOR EACH TEAM
    fixtures_combo = combine_fixture_stats(fixtures_stats, goals_stats, fixtures_master)
    team_fixture_stats = prepare_team_stats(fixtures_combo)
    team_fixture_features = compute_fixtures_features(team_fixture_stats)

    # MODEL FEATURES FOR EACH PLAYER

    # TARGET VARIABLE FOR EACH PLAYER

def get_goals_fixture_stats(fixtures_master):

    # extracting the data for halftime and fulltime goals
    goals_stats = fixtures_master.loc[:,['fixture_id','home_ht_goal','away_ht_goal','home_ft_goal','away_ft_goal']]
    goals_stats = goals_stats.melt(id_vars='fixture_id', var_name='var')
    goals_stats['value'] = goals_stats['value'].astype('float')
    goals_stats['team_side'] = goals_stats['var'].str.split('_', n = 1).map(lambda x: x[0])
    goals_stats['fixture_stat'] = goals_stats['var'].str.split('_', n = 1).map(lambda x: x[1])
    del goals_stats['var']

    goals_stats = pd.pivot_table(goals_stats, values='value',
                                 index=['fixture_id', 'fixture_stat'],
                                 columns='team_side', aggfunc='mean',
                                 fill_value=None, margins=False, dropna=False,
                                 margins_name='All', observed=False)
    goals_stats = goals_stats.reset_index(drop=False)
    goals_stats.columns = ['fixture_id', 'fixture_stat', 'away', 'home']

    return goals_stats


def get_detailed_fixture_stats(focus_source):
    # load the fixtures stats data
    fixtures_stats_file = os.path.join(_DATA_DIR, focus_source, 'extracted', 'fixtures_stats.csv')
    fixtures_stats = pd.read_csv(fixtures_stats_file, header=0, index_col=False)
    fixtures_stats['home'] = fixtures_stats['home'].str.replace('%', '').astype('float')
    fixtures_stats['away'] = fixtures_stats['away'].str.replace('%', '').astype('float')

    return fixtures_stats


def combine_fixture_stats(fixtures_stats, goals_stats, fixtures_master):

    # combining all the stats
    combo_fix_stats = fixtures_stats.append(goals_stats.loc[:, ['fixture_id', 'fixture_stat', 'home', 'away']])
    # merge the master onto the stats
    master_cols = ['event_date', 'fixture_id', 'league_id', 'home_team_id',
                   'away_team_id', 'home_team_name', 'away_team_name']
    fixtures_combo = combo_fix_stats.merge(fixtures_master.loc[:, master_cols],
                                           how='left', on='fixture_id')
    stats_col_order = ['fixture_id', 'event_date', 'league_id', 'home_team_id',
                       'home_team_name', 'away_team_id', 'away_team_name',
                       'fixture_stat', 'home', 'away']
    df_order = ['event_date', 'fixture_id', 'home_team_id', 'fixture_stat']
    fixtures_combo = fixtures_combo.loc[:, stats_col_order].reset_index(drop=True).sort_values(by=df_order)

    return fixtures_combo


def prepare_team_stats(fixtures_combo):

    # we are not interested in the home vs away structure anymore as each team
    # is evaluated independently, and then combined at the moment of the model prediction

    # looking at the home teams' stats only
    home_stats = fixtures_combo.loc[:, ['fixture_id','event_date','league_id',
                                       'home_team_id','home_team_name',
                                       'fixture_stat','home']]
    home_stats = home_stats.rename(columns={'home_team_id':'team_id', 'home_team_name':'team_name', 'home':'val'})
    # looking at the away teams' stats only
    away_stats = fixtures_combo.loc[:, ['fixture_id','event_date','league_id',
                                       'away_team_id','away_team_name',
                                       'fixture_stat','away']]
    away_stats = away_stats.rename(columns={'away_team_id':'team_id', 'away_team_name':'team_name', 'away':'val'})
    team_fixture_stats = home_stats.append(away_stats).reset_index(drop = True)0
    team_fixture_stats['event_date'] = pd.to_datetime(team_fixture_stats['event_date'], utc=True).dt.tz_localize(None)

    return team_fixture_stats


def compute_fixtures_features(team_fixture_stats):

    # rolling calculations for all the statistics except the goals
    stats_idx  = ~team_fixture_stats['fixture_stat'].isin(['ht_goal', 'ft_goal'])
    agg_col = ['team_id', 'team_name', 'fixture_stat']

    roll3 = team_fixture_stats.loc[stats_idx].groupby(agg_col)['val'].rolling(window=3, min_periods=2).mean()
    roll3 = roll3.reset_index().rename(columns = {'level_3' : 'prev_index'})
    roll3['feat_type'] = 'm3'

    roll5 = team_fixture_stats.loc[stats_idx].groupby(agg_col).rolling(window=5, min_periods=3).agg({"val": 'mean', "event_date": 'max'})
    roll5 = roll5.reset_index().rename(columns = {'level_3' : 'prev_index'})
    roll5['feat_type'] = 'm5'

    roll10 = team_fixture_stats.loc[stats_idx].groupby(agg_col).rolling(window=10, min_periods=5).agg({"val": "mean", "fixture_id": "tail"})
    roll10 = roll10.reset_index().rename(columns = {'level_3' : 'prev_index'})
    roll10['feat_type'] = 'm10'

    roll10sd = team_fixture_stats.loc[stats_idx].groupby(agg_col).rolling(window=10, min_periods=5).agg({"val": "std", "fixture_id": "tail"})
    roll10sd = roll10sd.reset_index().rename(columns = {'level_3' : 'prev_index'})
    roll10sd['feat_type'] = 'sd10'

    team_fixture_features = pd.concat([roll3, roll5, roll10, roll10sd])
    team_fixture_features['fixture_stat'] = team_fixture_features['fixture_stat'].replace(' ', '_', regex=True).str.lower()
    team_fixture_features['fxtr_feature'] = team_fixture_features['fixture_stat'] + '_' + team_fixture_features['feat_type']
    del team_fixture_features['fixture_stat']
    del team_fixture_features['feat_type']
    team_fixture_features = team_fixture_features.loc[~team_fixture_features['fixture_id'].isna()]

    team_fixture_features = team_fixture_features.merge(team_fixture_stats.loc[:, ['fixture_id','event_date']],
                                                        how='left',
                                                        left_index=True,
                                                        right_index=True)

    pivot_idx = ['fixture_id', 'event_date', 'team_id', 'team_name']
    test = pd.pivot_table(team_fixture_features.iloc[1:1000], values='val',
                                           index=pivot_idx,
                                           columns=['fxtr_feature'],
                                           aggfunc='count',
                                           fill_value=None, margins=False,
                                           dropna=False,
                                           margins_name='All', observed=False)
    test = test.reset_index(drop=False)

    return team_fixture_features
