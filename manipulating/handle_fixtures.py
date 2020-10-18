import os
import sys
import json
import pandas as pd
import datetime as dt
from dateutil import tz

from utils.utils_rapid_football import *

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def get_updated_fixture_master_stats(focus_source = 'rapid_football'):

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

    return None
