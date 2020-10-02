import os
import pandas as pd
import numpy as np

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main(focus_source = 'rapid_football'):

    # load the fixtures master
    master_file = os.path.join(_DATA_DIR, focus_source, 'masters', 'fixture_master.csv')
    fixtures_master = pd.read_csv(master_file, header=0, index_col=False)
    goals_stats = get_goals_fixture_stats(fixtures_master)
    fixtures_stats = get_detailed_fixture_stats(focus_source)
    assists = fixtures_stats.loc[fixtures_stats['fixture_stat'] == 'Assists', :]
    # RAW DATA FOR EACH TEAM
    fixtures_combo = combine_fixture_stats(fixtures_stats, goals_stats, fixtures_master)
    team_fixture_stats = prepare_team_stats(fixtures_combo)
    team_fixture_features = compute_fixtures_features(team_fixture_stats)
    team_feat_file = os.path.join(_DATA_DIR, 'target_features', 'team_features.csv')
    team_fixture_features.to_csv(team_feat_file, header=True, index=False)

    # inter = team_fixture_features.loc[team_fixture_features['team_name']=='Inter']
    # inter_feat_file = os.path.join(_DATA_DIR, 'inter_features.csv')
    # inter.to_csv(inter_feat_file, header=True, index=False)

    return None

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
    team_fixture_stats = home_stats.append(away_stats).reset_index(drop = True)
    team_fixture_stats['event_date'] = pd.to_datetime(team_fixture_stats['event_date'], utc=True).dt.tz_localize(None)

    return team_fixture_stats


def compute_fixtures_features(team_fixture_stats):

    # pre-computation data handling
    team_fixture_stats.sort_values(by = ['league_id','team_id','event_date','fixture_stat'], inplace = True)
    team_fixture_stats['val'] = team_fixture_stats['val'].fillna(0)

    # rolling calculations for all the statistics except the goals
    stats_idx  = ~team_fixture_stats['fixture_stat'].isin(['ht_goal', 'ft_goal'])
    agg_col = ['team_id', 'team_name', 'fixture_stat']

    roll3 = team_fixture_stats.loc[stats_idx].groupby(agg_col)['val'].rolling(window=3, min_periods=2).mean()
    roll3.index.rename('prev_index', level=3, inplace = True)
    roll3 = roll3.reset_index([0,1,2])
    roll3['feat_type'] = 'm3'

    roll5 = team_fixture_stats.loc[stats_idx].groupby(agg_col)['val'].rolling(window=5, min_periods=2).mean()
    roll5.index.rename('prev_index', level=3, inplace = True)
    roll5 = roll5.reset_index([0,1,2])
    roll5['feat_type'] = 'm5'

    roll10 = team_fixture_stats.loc[stats_idx].groupby(agg_col)['val'].rolling(window=10, min_periods=2).mean()
    roll10.index.rename('prev_index', level=3, inplace = True)
    roll10 = roll10.reset_index([0,1,2])
    roll10['feat_type'] = 'm10'

    roll10sd = team_fixture_stats.loc[stats_idx].groupby(agg_col)['val'].rolling(window=10, min_periods=2).std()
    roll10sd.index.rename('prev_index', level=3, inplace = True)
    roll10sd = roll10sd.reset_index([0,1,2])
    roll10sd['feat_type'] = 'sd10'

    # adding the half time and full time goal stats
    gol_s3 = team_fixture_stats.loc[~stats_idx].groupby(agg_col)['val'].rolling(window=3, min_periods=2).sum()
    gol_s3.index.rename('prev_index', level=3, inplace = True)
    gol_s3 = gol_s3.reset_index([0,1,2])
    gol_s3['feat_type'] = 's3'

    gol_s5 = team_fixture_stats.loc[~stats_idx].groupby(agg_col)['val'].rolling(window=5, min_periods=2).sum()
    gol_s5.index.rename('prev_index', level=3, inplace = True)
    gol_s5 = gol_s5.reset_index([0,1,2])
    gol_s5['feat_type'] = 's5'

    gol_s10 = team_fixture_stats.loc[~stats_idx].groupby(agg_col)['val'].rolling(window=10, min_periods=2).sum()
    gol_s10.index.rename('prev_index', level=3, inplace = True)
    gol_s10 = gol_s10.reset_index([0,1,2])
    gol_s10['feat_type'] = 's10'

    gol_sd10 = team_fixture_stats.loc[~stats_idx].groupby(agg_col)['val'].rolling(window=10, min_periods=2).std()
    gol_sd10.index.rename('prev_index', level=3, inplace = True)
    gol_sd10 = gol_sd10.reset_index([0,1,2])
    gol_sd10['feat_type'] = 'sd10'

    team_fixture_features = [roll3, roll5, roll10, roll10sd,
                             gol_s3, gol_s5, gol_s10, gol_sd10]
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
