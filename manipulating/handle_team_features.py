import os
import pandas as pd
import numpy as np

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def create_tm_features(focus_source, feat_windows):

    # load the fixtures master
    master_file = os.path.join(_DATA_DIR, focus_source, 'masters', 'fixture_master.csv')
    fixtures_master = pd.read_csv(master_file, header=0, index_col=False)
    goals_stats = _get_goals_fixture_stats(fixtures_master)
    fixtures_stats = _get_detailed_fixture_stats(focus_source)
    assists = fixtures_stats.loc[fixtures_stats['fixture_stat'] == 'Assists', :]
    # RAW DATA FOR EACH TEAM
    fixtures_combo = _combine_fixture_stats(fixtures_stats, goals_stats, fixtures_master)
    team_fixture_stats = _prepare_team_stats(fixtures_combo)
    mean_wind = feat_windows['mean_wind']
    sum_wind = feat_windows['sum_wind']
    sd_wind = feat_windows['sd_wind']
    team_fixture_features = _compute_fixtures_features(team_fixture_stats, mean_wind,
                                                      sum_wind, sd_wind)
    # adding the correct information about the round to each fixture
    calendar_file = os.path.join(_DATA_DIR, focus_source, 'masters', 'calendar_master.csv')
    calendar_data = pd.read_csv(calendar_file, header=0, index_col=False)
    keep_cols = ['league_id', 'fixture_id', 'match']
    team_fixture_features = team_fixture_features.merge(calendar_data.loc[:, keep_cols],
                                                        on=keep_cols[0:2],
                                                        how='left')
    team_feat_file = os.path.join(_DATA_DIR, 'target_features', 'team_features.csv')
    team_fixture_features.to_csv(team_feat_file, header=True, index=False)

    # inter = team_fixture_features.loc[team_fixture_features['team_name']=='Inter']
    # inter_feat_file = os.path.join(_DATA_DIR, 'inter_features.csv')
    # inter.to_csv(inter_feat_file, header=True, index=False)

    return None

def _get_goals_fixture_stats(fixtures_master):

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


def _get_detailed_fixture_stats(focus_source):
    # load the fixtures stats data
    fixtures_stats_file = os.path.join(_DATA_DIR, focus_source, 'extracted', 'fixtures_stats.csv')
    fixtures_stats = pd.read_csv(fixtures_stats_file, header=0, index_col=False)
    fixtures_stats['home'] = fixtures_stats['home'].str.replace('%', '').astype('float')
    fixtures_stats['away'] = fixtures_stats['away'].str.replace('%', '').astype('float')

    return fixtures_stats


def _combine_fixture_stats(fixtures_stats, goals_stats, fixtures_master):

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


def _prepare_team_stats(fixtures_combo):

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
    team_fixture_stats = home_stats.append(away_stats).reset_index(drop = True)
    team_fixture_stats['event_date'] = pd.to_datetime(team_fixture_stats['event_date'], utc=True).dt.tz_localize(None)

    return team_fixture_stats


def _compute_fixtures_features(team_fixture_stats, mean_wind, sum_wind, sd_wind):

    # pre-computation data handling
    team_fixture_stats.sort_values(by = ['league_id','team_id','event_date','fixture_stat'], inplace = True)
    team_fixture_stats['val'] = team_fixture_stats['val'].fillna(0)

    # rolling calculations for all the statistics except the goals
    stats_idx  = ~team_fixture_stats['fixture_stat'].isin(['ht_goal', 'ft_goal'])
    agg_col = ['team_id', 'team_name', 'fixture_stat']

    roll_mean_df = []
    for rw in mean_wind:
        single_roll_mean = team_fixture_stats.loc[stats_idx].groupby(agg_col)['val'].rolling(window=rw, min_periods=2).mean()
        single_roll_mean.index.rename('prev_index', level=3, inplace = True)
        single_roll_mean = single_roll_mean.reset_index([0,1,2])
        single_roll_mean['feat_type'] = 'm'+ str(rw)
        roll_mean_df.append(single_roll_mean)
    roll_mean_df = pd.concat(roll_mean_df)

    roll_sd_df = []
    for rw in sd_wind:
        single_roll_sd = team_fixture_stats.loc[stats_idx].groupby(agg_col)['val'].rolling(window=rw, min_periods=2).std()
        single_roll_sd.index.rename('prev_index', level=3, inplace = True)
        single_roll_sd = single_roll_sd.reset_index([0,1,2])
        single_roll_sd['feat_type'] = 'sd'+ str(rw)
        roll_sd_df.append(single_roll_sd)
    roll_sd_df = pd.concat(roll_sd_df)

    # adding the half time and full time goal stats
    roll_sum_df = []
    for rw in mean_wind:
        single_roll_sum = team_fixture_stats.loc[~stats_idx].groupby(agg_col)['val'].rolling(window=rw, min_periods=2).sum()
        single_roll_sum.index.rename('prev_index', level=3, inplace = True)
        single_roll_sum = single_roll_sum.reset_index([0,1,2])
        single_roll_sum['feat_type'] = 's'+ str(rw)
        roll_sum_df.append(single_roll_sum)
    roll_sum_df = pd.concat(roll_sum_df)

    team_fixture_features = [roll_mean_df, roll_sum_df, roll_sd_df]
    team_fixture_features = pd.concat(team_fixture_features)
    team_fixture_features['fixture_stat'] = team_fixture_features['fixture_stat'].replace(' ', '_', regex=True).str.lower()
    team_fixture_features['fxtr_feature'] = team_fixture_features['fixture_stat'] + '_' + team_fixture_features['feat_type']
    del team_fixture_features['fixture_stat']
    del team_fixture_features['feat_type']

    team_fixture_features = team_fixture_features.merge(team_fixture_stats.loc[:, ['fixture_id','event_date', 'league_id']],
                                                        how='left',
                                                        left_index=True,
                                                        right_index=True)
    pivot_idx = ['team_id', 'team_name', 'event_date', 'fxtr_feature', 'fixture_id', 'league_id']
    team_fixture_features = team_fixture_features.sort_values(by=pivot_idx)
    team_fixture_features = team_fixture_features.set_index(pivot_idx).unstack('fxtr_feature').reset_index()
    # flattening out the column index
    higher_levels = team_fixture_features.columns.get_level_values(0)
    higher_levels = [x for x in higher_levels if x != 'val']
    lower_levels = team_fixture_features.columns.get_level_values(1)
    lower_levels = [x for x in lower_levels if x != '']
    team_fixture_features.columns = higher_levels + lower_levels

    return team_fixture_features
