import os
import pickle
import json
import pandas as pd
import numpy as np

_BASE_DIR = '/home/costam/Documents'
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def get_ffdata_combined(select_seasons):

    # select_seasons = [2018, 2019, 2020]
    print('Loading team and fixtures masters.')
    team_map, fix_master = _get_ffdata_masters()
    print('Loading team and player features.')
    tm_feat, pl_feat = _get_ffdata_features()
    print('Loading player targets.')
    pl_fv = _get_ffdata_targets()
    # combining the data
    print('Combining the datasets.')
    full_dt = _combine_ffdata(team_map, fix_master, tm_feat, pl_feat, pl_fv, select_seasons)

    return full_dt


def _get_ffdata_masters():

    tm_master_file = os.path.join(_DATA_DIR, 'master', 'rapid_teamnames_map.csv')
    team_map = pd.read_csv(tm_master_file, header=0, index_col=False)

    fixture_master_file = os.path.join(_DATA_DIR, 'rapid_football/masters', 'fixture_master.csv')
    fix_master = pd.read_csv(fixture_master_file, header=0, index_col=False)

    return team_map, fix_master


def _get_ffdata_features():

    pl_feat_file = os.path.join(_DATA_DIR, 'target_features', 'players_features.csv')
    pl_feat = pd.read_csv(pl_feat_file, header=0, index_col=False)
    pl_feat = pl_feat.sort_values(['name', 'surname', 'season', 'match'])

    tm_feat_file = os.path.join(_DATA_DIR, 'target_features', 'team_features.csv')
    tm_feat = pd.read_csv(tm_feat_file, header=0, index_col=False)
    tm_feat = tm_feat.sort_values(['team_id', 'event_date'])

    # todo(Michele) : mapping of each Rapid API season to the year
    tm_feat.loc[tm_feat['league_id']==94, 'season'] = int(2018)
    tm_feat.loc[tm_feat['league_id']==891, 'season'] = int(2019)
    tm_feat.loc[tm_feat['league_id']==2857, 'season'] = int(2020)

    return tm_feat.drop_duplicates(), pl_feat.drop_duplicates()


def _get_ffdata_targets():

    pl_fv_file = os.path.join(_DATA_DIR, 'target_features', 'players_target.csv')
    pl_fv = pd.read_csv(pl_fv_file, header=0, index_col=False)
    pl_fv = pl_fv.sort_values(['name', 'surname', 'season', 'match'])
    # filling the fv with 6 if they have not played and centering fv on 6
    pl_fv['fwd_fv_scaled'] = pl_fv['fwd_fantavoto'].fillna(6.0) - 6

    return pl_fv.drop_duplicates()


def _combine_ffdata(team_map, fix_master, tm_feat, pl_feat, pl_fv, select_seasons):

    tm_dim = tm_feat.shape
    tm_counts = tm_feat.groupby('league_id')['match'].value_counts()
    pl_dim = pl_feat.shape

    # drop observations for games that have not happened yet / data has not been downloaded
    next_round_end = pd.Timestamp.today(tz='Europe/London') + pd.Timedelta(5, 'days')
    fix_master['event_date'] = pd.to_datetime(fix_master['event_date'], utc=True)
    fix_master = fix_master.loc[fix_master['event_date'] < next_round_end]

    pl_feat_ssn_idx = pl_feat['season'].isin(select_seasons)
    pl_fv_ssn_idx = pl_fv['season'].isin(select_seasons)

    # changing the structure of fixture master to find the teams more easily
    ext_fix_master = reshape_fixture_masters(fix_master)
    support_tm = tm_feat.loc[:, ['team_id', 'fixture_id', 'league_id', 'match','season']].drop_duplicates()
    ext_fix_master = ext_fix_master.merge(support_tm,
                                          left_on = ['league_id', 'fixture_id', 'core_team_id'],
                                          right_on = ['league_id', 'fixture_id', 'team_id'])
    # because we care about the next match opponent
    ext_fix_master['next_opponent'] = ext_fix_master.groupby(['core_team_id','season'])['opponent_team_id'].shift(-1)
    ext_fix_master.drop('team_id', axis=1, inplace=True)

    # getting the features split between team and player
    tm_remove = ['league_id', 'team_id', 'team_name', 'event_date', 'fixture_id', 'season','match']
    tm_cols = tm_feat.columns.values[~tm_feat.columns.isin(tm_remove)].tolist()
    pl_remove = ['team', 'name', 'surname', 'role', 'season', 'match']
    pl_cols = pl_feat.columns.values[~pl_feat.columns.isin(pl_remove)].tolist()

    # getting the home and away info for fixtures
    fix_cols = ['league_id','fixture_id','home_team_name','away_team_name', 'home_team_id', 'away_team_id']
    tm_feat = tm_feat.merge(fix_master.loc[:, fix_cols], how='inner',
                            on = ['league_id', 'fixture_id'])
    tm_feat.groupby('league_id')['match'].value_counts()
    tm_feat = tm_feat.merge(ext_fix_master.loc[:, ['core_team_id','season','match', 'next_opponent']],
                            left_on=['team_id','season','match'],
                            right_on=['core_team_id','season','match'],
                            how='left')

    # # core team is home team
    # core_home_idx = tm_feat['team_id'] == tm_feat['home_team_id']
    # tm_feat.loc[core_home_idx, 'opponent_name'] = tm_feat.loc[core_home_idx,'away_team_name']
    # tm_feat.loc[core_home_idx, 'opponent_id'] = tm_feat.loc[core_home_idx,'away_team_id']
    # # core team is away team
    # core_away_idx = tm_feat['team_id'] == tm_feat['away_team_id']
    # # core_away_ft = tm_feat.loc[core_away_idx,:].copy()
    # tm_feat.loc[core_away_idx, 'opponent_name'] = tm_feat.loc[core_away_idx,'home_team_name']
    # tm_feat.loc[core_away_idx, 'opponent_id'] = tm_feat.loc[core_away_idx,'home_team_id']
    # tm_feat.drop(['away_team_name','home_team_name', 'home_team_id', 'away_team_id'], axis=1, inplace=True)

    drop_cols = ['home_team_name', 'away_team_name', 'home_team_id', 'away_team_id' ,'core_team_id']
    tm_feat = tm_feat.rename(columns={'next_opponent':'opponent_id'}).drop(drop_cols, axis=1)

    # merging the opponents data onto the main df
    tm_oppo_feat = tm_feat.copy()
    tm_oppo_feat.drop(['team_name', 'event_date', 'fixture_id', 'opponent_id'],
                      axis=1, inplace=True)
    new_cols_order = ['team_id', 'league_id', 'match', 'season']
    new_cols_order = new_cols_order + tm_oppo_feat.columns.values[~tm_oppo_feat.columns.isin(new_cols_order)].tolist()
    tm_oppo_feat = tm_oppo_feat.loc[:,new_cols_order]
    new_col_names = 'op_'+tm_oppo_feat.columns[4:].values
    new_col_dict = dict(zip(tm_oppo_feat.columns[4:].values, new_col_names))
    tm_oppo_feat.rename(columns=new_col_dict, inplace=True)
    tm_feat = tm_feat.merge(tm_oppo_feat, how='left',
                            left_on = ['opponent_id', 'league_id', 'season', 'match'],
                            right_on = ['team_id', 'league_id', 'season', 'match'])
    tm_feat['team_id'] = tm_feat['team_id_x']
    tm_feat.drop(['team_id_x','team_id_y'], axis=1, inplace=True)

    # correct name of team coming from Gazzetta
    tm_feat = tm_feat.merge(team_map, left_on = 'team_name', right_on = 'team_name')
    tm_feat['team_name'] = tm_feat['team_gaz']
    tm_feat.drop(['team_gaz'], axis=1, inplace=True)

    # merging players target and players features
    full_dt = pl_fv.loc[pl_fv_ssn_idx,:].merge(pl_feat.loc[pl_feat_ssn_idx],
                                               on=['name', 'surname', 'season', 'match'],
                                               how ='left')
    full_dt['team'] = full_dt['team_x']
    full_dt.drop(['team_x', 'team_y'], axis=1, inplace=True)
    # filling NAs for player features
    pl_pivot = ['name', 'surname', 'season', 'match']
    full_dt = full_dt.sort_values(pl_pivot)
    # filling the players features
    full_dt.loc[:, pl_cols] = full_dt.groupby(pl_pivot[0:3])[pl_cols].fillna(method='ffill')
    # fixing role
    full_dt['role'] = full_dt.groupby(pl_pivot[0:3])['role'].fillna(method='ffill')
    full_dt['role'] = full_dt.groupby(pl_pivot[0:3])['role'].fillna(method='bfill')

    # adding the features for the player's team
    full_dt = full_dt.merge(tm_feat,
                            left_on=['team','season','match'],
                            right_on=['team_name','season', 'match'],
                            how='left')
    # filling the team features
    tm_pivot = ['team', 'season', 'match']
    full_dt = full_dt.sort_values(tm_pivot)
    full_dt.loc[:, tm_cols] = full_dt.groupby(tm_pivot[0:2])[tm_cols].fillna(method='ffill')
    oppo_tm_cols = ['op_' + x for x in tm_cols]
    full_dt.loc[:, oppo_tm_cols] = full_dt.groupby(tm_pivot[0:2])[oppo_tm_cols].fillna(method='ffill')

    # cleaning up the features data
    full_dt.columns.values
    drop_cols = ['team_id', 'opponent_id', 'team', 'team_name', 'event_date', 'fixture_id', 'league_id']
    full_dt.drop(drop_cols, axis=1, inplace=True)
    main_cols = ['name', 'surname', 'season', 'match', 'role']
    all_cols = main_cols + full_dt.columns.values[~full_dt.columns.isin(main_cols)].tolist()
    full_dt = full_dt.loc[:, all_cols]

    # dropping the features that have not enough data
    feat_na_check = full_dt.isna().sum()/full_dt.shape[0]
    feat_na_check = feat_na_check[feat_na_check>0.25]
    feat_na_check = feat_na_check.index.values[feat_na_check.index.values != 'fwd_fantavoto']
    if len(feat_na_check)>0:
        print(feat_na_check)
        full_dt.drop(feat_na_check, axis=1, inplace=True)

    # dropping the rows that have not enough data
    obs_na_check = full_dt.isna().sum(axis=1)/full_dt.shape[1]
    obs_na_check = obs_na_check[obs_na_check>0.3]
    if len(obs_na_check)>0:
        full_dt.drop(obs_na_check.index.values, axis=0, inplace=True)
        full_dt.reset_index(inplace=True, drop=True)

    full_dt['role'] = full_dt['role'].astype('category')
    role_dummy = pd.get_dummies(full_dt['role'])
    full_dt = pd.concat([full_dt, role_dummy], axis = 1)
    # full_dt = full_dt.drop('role', axis=1)

    return full_dt


def reshape_fixture_masters(fix_master):

    # we need to get from home-away pairs to team a - team b - round for all pairs
    keep_cols = ['event_date','fixture_id', 'league_id', 'home_team_id', 'away_team_id']
    home_fix_master = fix_master.loc[:, keep_cols].copy()
    home_fix_master.rename(columns = {'home_team_id':'core_team_id',
                                      'away_team_id':'opponent_team_id'}, inplace=True)
    home_fix_master['play_home'] = 'core'

    away_fix_master = fix_master.loc[:, keep_cols].copy()
    away_fix_master.rename(columns = {'home_team_id':'opponent_team_id',
                                      'away_team_id':'core_team_id'}, inplace=True)
    away_fix_master['play_home'] = 'oppo'
    away_fix_master = away_fix_master.loc[:, home_fix_master.columns]

    ext_fix_master = pd.concat([home_fix_master, away_fix_master])
    ext_fix_master.sort_values(by = ['league_id', 'core_team_id', 'event_date'], inplace= True)

    return ext_fix_master
