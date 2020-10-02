import os
import sys
import time
import random

import requests
from bs4 import BeautifulSoup
import re

import pandas as pd
import numpy as np


_BASE_DIR = '/home/costam/Documents'
_CODE_DIR = os.path.join(_BASE_DIR, 'fantacalcio/fanta_code')
_DATA_DIR = os.path.join(_BASE_DIR, 'fantacalcio/data')


def main():

    s = requests.session()
    # cookie_obj = requests.cookies.create_cookie(domain='',
    #                                             name='COOKIE_NAME',
    #                                             value='the cookie works')
    gaz_cookie = {
        "version":0,
        "name":'COOKIE_NAME',
        "value":'true',
        "port":None,
        # "port_specified":False,
        "domain":'https://www.gazzetta.it/',
        # "domain_specified":False,
        # "domain_initial_dot":False,
        "path":'/',
        # "path_specified":True,
        "secure":False,
        "expires":None,
        "discard":True,
        "comment":None,
        "comment_url":None,
        "rest":{'HttpOnly': None},
        "rfc2109":False
        }
    s.cookies.set(**gaz_cookie)

    # seasons = [2020, 2019, 2018, 2017]
    seasons = [2020]
    for season in seasons:
        for matchday in range(1, 39):
            # matchday = 1
            time.sleep(random.randint(15, 35))
            req = send_gaz_request(s, season, matchday)
            fv_df = extract_gaz_data(req)
            outfile = 'a' + str(season) + '_m'+ str(matchday) + '.csv'
            if fv_df.shape[0]>1:
                fv_df.to_csv(os.path.join(_DATA_DIR, 'fantavoti', outfile), header=True, index=False)

    return None


def send_gaz_request(s, season, matchday):

    main_url = 'https://www.gazzetta.it/calcio/fantanews/voti/'
    # 'serie-a-2019-20/giornata-38'
    torneo = 'serie-a'
    stagione = str(season) + '-' + str(int(str(season)[2:])+1)
    giornata = 'giornata-' + str(matchday)
    url = main_url + torneo + '-' + stagione + '/' + giornata
    # headers
    headers = create_req_headers()
    # get request
    req = s.get(url, headers=headers)
    # safety checks
    status_code = req.status_code
    if status_code == requests.codes.ok:
        return req
    elif status_code == requests.codes.bad:
        sys.exit('Invalid request. Check parameters.')
    elif status_code == requests.codes.forbidden:
        sys.exit('This resource is restricted')
    elif status_code == requests.codes.not_found:
        sys.exit('This resource does not exist. Check parameters')
    elif status_code == requests.codes.too_many_requests:
        sys.exit('You have exceeded your allowed requests per minute/day')

    # try:
    #     # something
    # except Exception as e:
    #     # If the download for some reason fails (ex. 404) the script will continue downloading
    #     # the next article.
    #     print(e)
    #     print("continuing...")
    #     continue

    return req


def extract_gaz_data(req):

    # HTML elemenets:
    # ul : unordered list || li : ordered list
    soup = BeautifulSoup(req.content, 'html.parser')

    # table header : V, G, A, R, RS, AG, AM, ES, FV
    # Voto, Goal, Assist, Rigore, RigoreSbagliato, Autogol, Ammonizione, Espulsione, Fantavoto
    # RT : rigori tirati
    # RR : Rigori riusciti
    # RS : Rigori Sbagliati
    # RP : Rigori Parati (portiere)

    fv_columns = ['team', 'name', 'surname', 'role', 'voto', 'gol', 'assist',
                  'rigore', 'rig_sbagliato', 'autogol', 'ammonizione', 'espulsione',
                  'fantavoto']
    fv_df = pd.DataFrame(columns=fv_columns)

    voti_squadre = soup.find_all('ul', attrs={'class': 'magicTeamList'})
    for vs in voti_squadre[0:20]:
        nome_squadra = vs.find('span', attrs={'class': 'teamNameIn'}).text
        voti_giocatori = vs.find_all('li')
        # first list component is the header of the table so we skip
        for vg in voti_giocatori[1:]:
            player, role = extract_html_player_info(vg)
            stats = extract_html_player_stats(vg)
            new_row = {'team' : nome_squadra,
                       'name' : player[0],
                       'surname' : player[1],
                       'role' : role,
                       'voto' : stats[0],
                       'gol' : stats[1],
                       'assist' : stats[2],
                       'rigore' : stats[3],
                       'rig_sbagliato' : stats[4],
                       'autogol' : stats[5],
                       'ammonizione' : stats[6],
                       'espulsione' : stats[7],
                       'fantavoto' : stats[8]}
            fv_df = fv_df.append(new_row, ignore_index=True)

    return fv_df


def create_req_headers():

    desktop_agents = [
    'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2224.3 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2225.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.4; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2225.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.1 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2227.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2226.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.93 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36'
    ]

    string_accept = 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    headers = {'User-Agent': random.choice(desktop_agents),
               'Accept': string_accept}

    return headers


def find_html_player_data(tag):
    # function to get all the data for a player without splitting the search into
    # 'inParameter Vparameter', 'inParameter FVParameter', 'inParameter',
    if tag.has_attr('class'):
        if tag.get('class')[0] == 'inParameter':
            return tag


def extract_html_player_info(vg):

    # vg is the HTML tree containing the player informations
    player = vg.find('span', attrs={'class': 'playerNameIn'}).find('a').get('href')
    # print(player)
    play_str_2019 = 'https://www.gazzetta.it/calcio/giocatori/'
    play_str_2018 = 'https://www.gazzetta.it/calcio/fantanews/statistiche/'
    # https://www.gazzetta.it/calcio/fantanews/statistiche/serie-a-2018-19/duvan_zapata_928

    if play_str_2019 in player:
        player = player.replace(play_str_2019, '').split('/')[0].split('-')
    elif play_str_2018 in player:
        player = player.replace(play_str_2018, '').split('/')[1].split('_')[:-1]

    if len(player) ==1:
        player = ['', player[0]]
    elif len(player) >2:
        player = [player[0], ' '.join(player[1:])]

    print(player)
    role = vg.find('span', attrs={'class': 'playerRole'}).text

    return player, role


def extract_html_player_stats(vg):

    # extract the HMTL components based on attributes
    stats = vg.find_all(find_html_player_data)
    # cleanup strings and from string to integer
    stats = [clean_player_data(x) for x in stats]
    return stats


def clean_player_data(string_data):

    delete_string = re.compile(r'[\n\r\t ]')
    string_data = delete_string.sub('', string_data.get_text())

    if string_data == '-':
        return np.NaN
    else:
         return float(string_data)
