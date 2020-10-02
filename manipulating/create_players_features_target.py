import os
import pandas as pd
import numpy as np
# import datetime as dt
# from dateutil import tz

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main():

    # unique dataframe from single round-season data
    gaz_fv = extract_gaz_data()
    # work on players master
    gaz_players = get_players_master(gaz_fv)
    # add the correct name-surnames after manual process
    gaz_players, gaz_fv = clean_players_name(gaz_players, gaz_fv)
    # work on fantavoti history and players stats
    players_features = build_players_features(gaz_fv)
    players_feat_file = os.path.join(_DATA_DIR, 'target_features', 'players_features.csv')
    players_features.to_csv(players_feat_file, header=True, index=False)
    # players fantavoto as model target variable
    players_target = build_players_target(gaz_fv)
    players_trgt_file = os.path.join(_DATA_DIR, 'target_features', 'players_target.csv')
    players_target.to_csv(players_trgt_file, header=True, index=False)

def extract_gaz_data():

    gaz_fv_files = os.listdir(os.path.join(_DATA_DIR, 'fantavoti'))
    gaz_fv = []
    for f in gaz_fv_files:
        single_round_fv = pd.read_csv(os.path.join(_DATA_DIR, 'fantavoti', f), header=0, index_col=False)
        single_round_fv['season'] = int(str(f)[1:5])
        single_round_fv['match'] = int(str(f)[7:(len(f)-4)])
        if 'Unnamed: 0' in single_round_fv.columns.values:
            del single_round_fv['Unnamed: 0']
        gaz_fv.append(single_round_fv)
    gaz_fv = pd.concat(gaz_fv, ignore_index=True)
    gaz_fv['name'] = gaz_fv['name'].fillna('_unknown_')

    return gaz_fv


def get_players_master(gaz_fv):

    # collapse the data to players and teams only
    agg_dict = {'season' : [np.min, np.max], 'match' : [np.min, np.max]}
    gaz_players = gaz_fv.groupby(['team','name','surname', 'role']).agg(agg_dict)
    # multilevels index and columns so need to flatten it
    gaz_players.columns = ['_'.join(col).strip() for col in gaz_players.columns.values]
    gaz_players = gaz_players.reset_index()
    gaz_players = gaz_players.sort_values(by=['team', 'role', 'surname'], ascending=True)

    print('Saving the players master to file.')
    gaz_players.to_csv(os.path.join(_DATA_DIR, 'master', 'gaz_players_master.csv'),header=True, index=False)

    return gaz_players


def clean_players_name(gaz_players, gaz_fv):

    raw_filename = os.path.join(_DATA_DIR, 'master', 'gaz_players_correction.csv')
    manual_filename = os.path.join(_DATA_DIR, 'master', 'gaz_players_correction_manual.csv')

    idx_two_surnames = gaz_players['surname'].str.split(' ').apply(len) >1
    name_player_checks =  gaz_players.loc[idx_two_surnames, ['name','surname']].drop_duplicates()
    name_player_checks.to_csv(raw_filename, header=True, index=False)
    print('Adding the correct name-surname combinations.')
    cleaned_names = pd.read_csv(manual_filename, header=0, index_col=False)
    gaz_players = gaz_players.merge(cleaned_names, how='left',
                                    on=['name','surname'])
    correction_idx = ~(gaz_players['name_correct'].isna() & gaz_players['surname_correct'].isna())
    gaz_players.loc[correction_idx, 'name'] = gaz_players.loc[correction_idx, 'name_correct']
    gaz_players.loc[correction_idx, 'surname'] = gaz_players.loc[correction_idx, 'surname_correct']
    del gaz_players['name_correct']
    del gaz_players['surname_correct']

    gaz_fv = gaz_fv.merge(cleaned_names, how='left',
                                    on=['name','surname'])
    correction_idx = ~(gaz_fv['name_correct'].isna() & gaz_fv['surname_correct'].isna())
    gaz_fv.loc[correction_idx, 'name'] = gaz_fv.loc[correction_idx, 'name_correct']
    gaz_fv.loc[correction_idx, 'surname'] = gaz_fv.loc[correction_idx, 'surname_correct']
    del gaz_fv['name_correct']
    del gaz_fv['surname_correct']

    return gaz_players, gaz_fv


def build_players_features(gaz_fv):

    long_gaz_fv = gaz_fv.copy()
    # handling the voto and fantavoto data and re-normalising
    long_gaz_fv['voto2'] = long_gaz_fv['voto'].fillna(6.)
    long_gaz_fv['fantavoto2'] = long_gaz_fv['fantavoto'].fillna(6.)
    long_gaz_fv['voto2'] = long_gaz_fv['voto'] -6
    long_gaz_fv['fantavoto2'] = long_gaz_fv['fantavoto'] -6
    long_gaz_fv['presenza'] = ~long_gaz_fv['voto'].isna()
    del long_gaz_fv['fantavoto']
    del long_gaz_fv['voto']
    # reshape to long format
    long_gaz_fv = pd.melt(long_gaz_fv, id_vars = ['team','name','surname','role','season','match'],
                          var_name='gaz_stat', value_name='val')
    long_gaz_fv = long_gaz_fv.sort_values(by = ['role', 'surname','name', 'season','match'])
    # filling the gaps
    idx = long_gaz_fv['gaz_stat'].isin(['presenza'])
    long_gaz_fv.loc[~idx, 'val'] = long_gaz_fv.loc[~idx, 'val'].fillna(0)

    # rolling calculations for all the statistics except the goals
    stats_idx  = ~long_gaz_fv['gaz_stat'].isin(['presenza', 'voto2', 'fantavoto2'])
    agg_col = ['name',	'surname', 'gaz_stat']

    roll3 = long_gaz_fv.loc[stats_idx].groupby(agg_col)['val'].rolling(window=3, min_periods=2).sum()
    roll3.index.rename('prev_index', level=3, inplace = True)
    roll3 = roll3.reset_index([0,1,2])
    roll3['feat_type'] = 's3'

    roll5 = long_gaz_fv.loc[stats_idx].groupby(agg_col)['val'].rolling(window=5, min_periods=2).sum()
    roll5.index.rename('prev_index', level=3, inplace = True)
    roll5 = roll5.reset_index([0,1,2])
    roll5['feat_type'] = 's5'

    roll10 = long_gaz_fv.loc[stats_idx].groupby(agg_col)['val'].rolling(window=10, min_periods=2).sum()
    roll10.index.rename('prev_index', level=3, inplace = True)
    roll10 = roll10.reset_index([0,1,2])
    roll10['feat_type'] = 's10'

    roll10sd = long_gaz_fv.loc[stats_idx].groupby(agg_col)['val'].rolling(window=10, min_periods=2).std()
    roll10sd.index.rename('prev_index', level=3, inplace = True)
    roll10sd = roll10sd.reset_index([0,1,2])
    roll10sd['feat_type'] = 'sd10'

    # adding rolling count of matches
    pres_idx = long_gaz_fv['gaz_stat'] == 'presenza'
    count10 = long_gaz_fv.loc[pres_idx,].groupby(agg_col)['val'].rolling(window=10, min_periods=1).sum()
    count10.index.rename('prev_index', level=3, inplace = True)
    count10 = count10.reset_index([0,1,2])
    count10['feat_type'] = 'cnt10'

    # adding the fantavoto and voto stats
    voto_idx = long_gaz_fv['gaz_stat'].isin(['voto2', 'fantavoto2'])
    voto3 = long_gaz_fv.loc[voto_idx].groupby(agg_col)['val'].rolling(window=3, min_periods=1).mean()
    voto3.index.rename('prev_index', level=3, inplace = True)
    voto3 = voto3.reset_index([0,1,2])
    voto3['feat_type'] = 'm3'

    voto5 = long_gaz_fv.loc[voto_idx].groupby(agg_col)['val'].rolling(window=5, min_periods=1).mean()
    voto5.index.rename('prev_index', level=3, inplace = True)
    voto5 = voto5.reset_index([0,1,2])
    voto5['feat_type'] = 'm5'

    voto10 = long_gaz_fv.loc[voto_idx].groupby(agg_col)['val'].rolling(window=10, min_periods=1).mean()
    voto10.index.rename('prev_index', level=3, inplace = True)
    voto10 = voto10.reset_index([0,1,2])
    voto10['feat_type'] = 'm10'

    voto10sd = long_gaz_fv.loc[voto_idx].groupby(agg_col)['val'].rolling(window=10, min_periods=2).std()
    voto10sd.index.rename('prev_index', level=3, inplace = True)
    voto10sd = voto10sd.reset_index([0,1,2])
    voto10sd['feat_type'] = 'sd10'

    players_features = [roll3, roll5, roll10, roll10sd, count10,
                             voto3, voto5, voto10, voto10sd]
    players_features = pd.concat(players_features)
    players_features.loc[players_features['gaz_stat'] == 'voto2', 'gaz_stat'] = 'voto'
    players_features.loc[players_features['gaz_stat'] == 'fantavoto2', 'gaz_stat'] = 'fantavoto'
    players_features['gaz_stat'] = players_features['gaz_stat'].replace(' ', '_', regex=True).str.lower()
    players_features['gaz_stat'] = players_features['gaz_stat'] + '_' + players_features['feat_type']
    del players_features['feat_type']

    keep_cols = ['team','role','season','match']
    players_features = players_features.merge(long_gaz_fv.loc[:, keep_cols],
                                              how='left',
                                              left_index=True,
                                              right_index=True)
    pivot_idx = ['team', 'name', 'surname', 'role','season','match', 'gaz_stat']

    players_features['role'] = players_features['role'].astype('category')
    players_features = players_features.sort_values(by=pivot_idx)
    players_features = players_features.set_index(pivot_idx).unstack('gaz_stat').reset_index()
    # flattening out the column index
    higher_levels = players_features.columns.get_level_values(0)
    higher_levels = [x for x in higher_levels if x != 'val']
    lower_levels = players_features.columns.get_level_values(1)
    lower_levels = [x for x in lower_levels if x != '']
    players_features.columns = higher_levels + lower_levels

    return players_features


def build_season_skeleton(gaz_fv):

    gaz_all_seasons = []
    for s in sorted(gaz_fv['season'].unique()):
        # s = 2019
        season_players = gaz_fv.loc[gaz_fv['season'] == s, ['name', 'surname']].drop_duplicates()
        season_players['gaz_pl_id'] = season_players.index.values + 1
        matches = np.arange(1,39)
        season_index = pd.MultiIndex.from_product([season_players['gaz_pl_id'], matches], names = ["gaz_pl_id", "match"])
        season_skel = pd.DataFrame(index = season_index).reset_index()
        season_skel = season_skel.merge(season_players, on = 'gaz_pl_id', how='left')
        season_skel['season'] = s
        del season_skel['gaz_pl_id']
        gaz_all_seasons.append(season_skel)

    gaz_all_seasons = pd.concat(gaz_all_seasons)
    gaz_all_seasons.reset_index(drop=True, inplace=True)

    return gaz_all_seasons


def build_players_target(gaz_fv):

    # getting all the matches for each player available in that season
    gaz_all_seasons = build_season_skeleton(gaz_fv)
    # merging the gazzetta marks
    keep_cols = ['name', 'surname','season', 'match', 'team','fantavoto']

    players_target = gaz_all_seasons.merge(gaz_fv.loc[:, keep_cols],how='left',
                                            on=['name', 'surname','season', 'match'])
    players_target = players_target.sort_values(by=['name', 'surname', 'season', 'match'])
    pivot_cols = ['name', 'surname','season']
    # filling the team name if they have not received any fv in that match
    players_target['team'] = players_target.groupby(pivot_cols)['team'].fillna(method='ffill')
    # filling the team col on the first matches of the season if they have not played
    players_target['team'] = players_target.groupby(pivot_cols)['team'].fillna(method='backfill', axis=0)
    # shifting the fantavoto one match earlier to have the Forward Realised Fantavoto
    players_target['fwd_fantavoto'] = players_target.groupby(pivot_cols)['fantavoto'].shift(-1)

    return players_target.loc[:, ['season','match','team','name', 'surname', 'fwd_fantavoto']]
