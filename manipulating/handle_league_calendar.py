import os
import sys
import json
import pandas as pd
import datetime as dt
from dateutil import tz

_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def get_updated_calendar(focus_source = 'rapid_football'):

    # loading the previously built master for all matches (fixtures)
    calendar_file = os.path.join(_DATA_DIR, focus_source, 'masters', 'calendar_master.csv')
    calendar_data = _get_fixture_calendar(calendar_file)
    calendar_data = _update_calendar(calendar_data, focus_source)
    calendar_data.to_csv(calendar_file, header=True, index=False)

    return calendar_data


def _get_fixture_calendar(calendar_file):

    # creating the file master if not available
    if not os.path.isfile(calendar_file):
        calendar_cols = ['fixture_id', 'league_id', 'event_date', 'match', 'status', 'statusShort']
        calendar_data = pd.DataFrame(columns=calendar_cols)
        calendar_data.to_csv(calendar_file, header=True, index=False)

    calendar_data = pd.read_csv(calendar_file, header=0, index_col=False)
    return calendar_data


def _update_calendar(calendar_data, focus_source):

    calendar_dir = os.path.join(_DATA_DIR, focus_source, 'leagueround')
    fixture_rounds = os.listdir(calendar_dir)
    fixture_rounds = sorted([x for x in fixture_rounds if x.endswith('.json')])
    for fr in fixture_rounds:
        # fr = fixture_rounds[0]
        with open(os.path.join(calendar_dir, fr), 'r') as inputfile:
            raw_data = json.load(inputfile)
        round_id = int(fr.split('_')[1].replace('round-', ''))
        raw_data = pd.DataFrame.from_dict(raw_data)
        calendar_cols = ['fixture_id', 'league_id', 'event_date', 'status', 'statusShort']
        raw_data = raw_data.loc[:, calendar_cols]
        raw_data['match'] = round_id
        calendar_data = pd.concat([calendar_data, raw_data])

    # correctly setting the date and time
    calendar_data['event_date'] = pd.to_datetime(calendar_data['event_date'], utc=True)
    calendar_data = calendar_data.set_index('event_date', drop=True).tz_convert('Europe/London').reset_index()

    return calendar_data
