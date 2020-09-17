import os
import pandas as pd
import numpy as np
# import datetime as dt
# from dateutil import tz

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main():

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
    # collapse the data to players and teams only
    agg_dict = {'season' : [np.min, np.max], 'match' : [np.min, np.max]}
    gaz_players = gaz_fv.groupby(['team','name','surname', 'role']).agg(agg_dict)
    # multilevels index and columns so need to flatten it
    gaz_players.columns = ['_'.join(col).strip() for col in gaz_players.columns.values]
    gaz_players = gaz_players.reset_index()
    gaz_players = gaz_players.sort_values(by=['team', 'role', 'surname'], ascending=True)

    gaz_players.to_csv(os.path.join(_DATA_DIR, 'master', 'gaz_players_master.csv'),header=True, index=False)

    surname_two_strings = gaz_players['surname'].str.split(' ')
    surname_two_strings = [x for x in surname_two_strings if len(x) >1]

    # erroneous splits
ali	kadhim adnan
jose	luis palomino
jean	claude billong
costa	marques guilherme
