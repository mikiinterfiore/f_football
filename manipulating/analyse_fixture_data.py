import os
import sys
import json
import pandas as pd
import datetime as dt
from dateutil import tz

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main(focus_source = 'rapid_football'):

    # loading the previously built master for all matches (fixtures)
    master_file = os.path.join(_DATA_DIR, focus_source, 'masters', 'fixture_master.csv')
    fixtures_master = get_fixture_master(focus_source, master_file)
    # update with new content if available
    fixture_dir = os.path.join(_DATA_DIR, focus_source, 'fixtureid')
    fixtures_master = update_fixture_master(fixtures_master, fixture_dir)
    fixtures_master.to_csv(master_file, header=True, index=False)

    # get the statistics for each fixture
    fixtures_stats_file = os.path.join(_DATA_DIR, focus_source, 'extracted', 'fixtures_stats.csv')
    fixtures_stats = get_fixture_stats_data(focus_source, fixtures_stats_file)
    fixtures_stats = update_fixtures_stats(fixtures_stats, fixtures_master,
                                           fixtures_stats_file, focus_source)
    fixtures_stats.to_csv(fixtures_stats_file, header=True, index=False)


def get_fixture_stats_data(focus_source, fixtures_stats_file):

    if not os.path.isfile(fixtures_stats_file):
        fix_stats_cols = ['fixture_id', 'fixture_stat', 'home', 'away']
        fixtures_stats = pd.DataFrame(columns=fix_stats_cols)
        fixtures_stats.to_csv(fixtures_stats_file, header=True, index=False)

    fixtures_stats = pd.read_csv(fixtures_stats_file, header=0, index_col=False)

    return fixtures_stats


def update_fixtures_stats(fixtures_stats, fixtures_master, fixtures_stats_file, focus_source):

    for sf in fixtures_master['fixture_id'].unique():
        sf_file = os.path.join(_DATA_DIR, focus_source, 'statisticsfixture',
                                'fixtureid-'+str(sf)+'_.json')
        if os.path.isfile(sf_file):
            sf_data = pd.read_json(sf_file, orient = 'columns')
            sf_data['fixture_id'] = sf
            fixtures_stats = fixtures_stats.append(sf_data)

    return fixtures_stats.drop_duplicates()


def add_fixture_data(fixtures_master, sf):

    new_row = {
    'fixture_id' : int(sf['fixture_id']),
    'league_id' : int(sf['league_id']),
    # 'event_date' : dt.datetime.strptime(sf['event_date'],'%Y-%m-%dT%H:%M:S+%z'),
    'event_date' : pd.to_datetime(sf['event_timestamp'], unit='s'),
    'referee' : sf['referee'],
    'home_team_id' : int(sf['homeTeam']['team_id']),
    'home_team_name' : sf['homeTeam']['team_name'],
    'away_team_id' : int(sf['awayTeam']['team_id']),
    'away_team_name' : sf['awayTeam']['team_name'],
    'home_ht_goal' : int(sf['score']['halftime'].split('-')[0]),
    'away_ht_goal' : int(sf['score']['halftime'].split('-')[1]),
    'home_ft_goal' : int(sf['goalsHomeTeam']),
    'away_ft_goal' : int(sf['goalsAwayTeam'])
    }

    fixtures_master = fixtures_master.append(new_row, ignore_index=True)

    return fixtures_master


def get_fixture_master(focus_source, master_file):

    # creating the file master if not available
    if not os.path.isfile(master_file):
        fix_master_cols = ['fixture_id', 'league_id', 'event_date', 'referee',
                           'home_team_id', 'away_team_id', 'home_team_name',
                           'away_team_name', 'home_ht_goal', 'away_ht_goal',
                           'home_ft_goal', 'away_ft_goal']
        fixtures_master = pd.DataFrame(columns=fix_master_cols)
        fixtures_master.to_csv(master_file, header=True, index=False)

    fixtures_master = pd.read_csv(master_file, header=0, index_col=False)
    return fixtures_master


def update_fixture_master(fixtures_master, fixture_dir):

    all_team_fixtures = os.listdir(fixture_dir)
    all_team_fixtures = [x for x in all_team_fixtures if x.endswith('.json')]
    for tf in all_team_fixtures:
        with open(os.path.join(fixture_dir, tf), 'r') as inputfile:
            raw_data = json.load(inputfile)
        for sf in raw_data:
            if sf['fixture_id'] not in fixtures_master['fixture_id'].unique():
                fixtures_master = add_fixture_data(fixtures_master, sf)

    # correctly setting the date and time
    fixtures_master['event_date'] = pd.to_datetime(fixtures_master['event_date'], utc=True)
    fixtures_master = fixtures_master.set_index('event_date', drop=True).tz_convert('Europe/London').reset_index()

    return fixtures_master
